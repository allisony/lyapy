"""Audit real STIS G140M Ly-alpha exposures before fitting them.

This is deliberately a data-validation workflow, not a posterior fit. It
answers the questions that must be settled first: which exposures have usable
Ly-alpha pixels, where geocoronal Ly-alpha falls after the pipeline's
heliocentric correction, whether the reported errors resemble the local
scatter, and whether separate visits can plausibly share one line amplitude.

The default paths reproduce the local TOI-178 validation dataset. The FITS
file and generated products stay under ``user_data`` and are ignored by Git::

    python tools/validate_g140m_observations.py

Use another directory without changing the script::

    python tools/validate_g140m_observations.py path/to/x1d/files \
        --target-name "My target" --stellar-rv 12.3 --no-coadded
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.observations import (  # noqa: E402
    SPEED_OF_LIGHT_KM_S,
    STIS_DQ_FLAGS,
    load_stis_x1d,
    stis_airglow_wavelength,
)


LYA_REST_WAVELENGTH = 1215.67
DEFAULT_FIT_WINDOW = (1214.4, 1217.2)
DEFAULT_RED_WING_WINDOW = (1215.95, 1216.55)
DEFAULT_NOISE_WINDOWS = ((1214.4, 1215.2), (1216.7, 1217.2))


def _window_mask(wavelength, window):
    lower, upper = window
    return (wavelength >= lower) & (wavelength <= upper)


def _union_mask(wavelength, windows):
    mask = np.zeros(wavelength.shape, dtype=bool)
    for window in windows:
        mask |= _window_mask(wavelength, window)
    return mask


def _integrated_flux(spectrum, window):
    mask = _window_mask(spectrum.wavelength, window) & spectrum.quality_mask
    if np.count_nonzero(mask) < 2:
        return None, None, int(np.count_nonzero(mask))
    wavelength = spectrum.wavelength[mask]
    flux = spectrum.flux[mask]
    error = spectrum.error[mask]
    widths = np.gradient(wavelength)
    value = float(np.sum(flux * widths))
    uncertainty = float(np.sqrt(np.sum((error * widths) ** 2)))
    return value, uncertainty, int(wavelength.size)


def assess_exposure(
    spectrum,
    *,
    fit_window,
    red_wing_window,
    noise_windows,
    airglow_half_width,
    maximum_extraction_offset,
):
    """Return a JSON-ready, per-exposure scientific audit."""
    wavelength = spectrum.wavelength
    fit_mask = _window_mask(wavelength, fit_window)
    usable_fit_mask = fit_mask & spectrum.quality_mask
    fit_pixels = int(np.count_nonzero(fit_mask))
    usable_fit_pixels = int(np.count_nonzero(usable_fit_mask))

    velocity = spectrum.metadata.get("heliocentric_velocity_km_s")
    airglow_center = (
        None if velocity is None else stis_airglow_wavelength(velocity)
    )
    airglow_window = (
        None
        if airglow_center is None
        else (
            airglow_center - airglow_half_width,
            airglow_center + airglow_half_width,
        )
    )

    flag_counts = {}
    for flag, description in STIS_DQ_FLAGS.items():
        count = int(
            np.count_nonzero(
                fit_mask & ((spectrum.data_quality & flag) != 0)
            )
        )
        if count:
            flag_counts[str(flag)] = {
                "pixels": count,
                "condition": description,
            }

    noise_mask = (
        _union_mask(wavelength, noise_windows) & spectrum.quality_mask
    )
    if airglow_window is not None:
        noise_mask &= ~_window_mask(wavelength, airglow_window)
    if np.count_nonzero(noise_mask) >= 2:
        noise_flux = spectrum.flux[noise_mask]
        residual_rms = float(
            np.sqrt(np.mean((noise_flux - np.median(noise_flux)) ** 2))
        )
        median_error = float(np.median(spectrum.error[noise_mask]))
        rms_to_error = (
            None if median_error <= 0.0 else residual_rms / median_error
        )
    else:
        residual_rms = median_error = rms_to_error = None

    red_flux, red_error, red_pixels = _integrated_flux(
        spectrum,
        red_wing_window,
    )
    extraction_offset = spectrum.metadata.get("extraction_offset_pixels")
    suspicious_extraction = (
        extraction_offset is not None
        and abs(extraction_offset) > maximum_extraction_offset
    )
    accepted = usable_fit_pixels >= 2 and not suspicious_extraction
    if suspicious_extraction:
        reason = (
            "automatic trace location moved "
            f"{extraction_offset:.1f} pixels; inspect the 2-D product and "
            "re-extract before scientific use"
        )
    elif usable_fit_pixels < 2:
        reason = "fewer than two Ly-alpha fit pixels survive serious DQ flags"
    else:
        reason = "usable"
    return {
        "filename": spectrum.metadata.get("filename"),
        "rootname": spectrum.metadata.get("rootname"),
        "observation_set": spectrum.metadata.get("observation_set"),
        "date_obs": spectrum.metadata.get("date_obs"),
        "time_obs": spectrum.metadata.get("time_obs"),
        "exposure_time_seconds": spectrum.metadata.get(
            "exposure_time_seconds"
        ),
        "accepted_for_lya": accepted,
        "assessment": reason,
        "automatic_extraction_suspicious": suspicious_extraction,
        "extraction_diagnostics": {
            "a2center_pixels": spectrum.metadata.get(
                "extraction_center_a2_pixels"
            ),
            "offset_pixels": extraction_offset,
            "max_search_pixels": spectrum.metadata.get(
                "extraction_max_search_pixels"
            ),
            "box_height_pixels": spectrum.metadata.get(
                "extraction_box_height_pixels"
            ),
            "offset_warning_threshold_pixels": maximum_extraction_offset,
            "threshold_provenance": (
                "conservative workflow heuristic, not an official STScI "
                "quality threshold"
            ),
        },
        "fit_pixels": fit_pixels,
        "usable_fit_pixels": usable_fit_pixels,
        "usable_fit_fraction": (
            None if fit_pixels == 0 else usable_fit_pixels / fit_pixels
        ),
        "fit_window_dq_flags": flag_counts,
        "heliocentric_velocity_km_s": velocity,
        "predicted_airglow_center_angstrom": airglow_center,
        "airglow_exclusion_window_angstrom": (
            None if airglow_window is None else list(airglow_window)
        ),
        "red_wing": {
            "window_angstrom": list(red_wing_window),
            "integrated_flux": red_flux,
            "uncertainty": red_error,
            "pixels": red_pixels,
        },
        "local_noise_diagnostic": {
            "windows_angstrom": [list(window) for window in noise_windows],
            "residual_rms": residual_rms,
            "median_pipeline_error": median_error,
            "rms_to_median_error": rms_to_error,
            "interpretation": (
                "diagnostic only; line wings and isolated features can inflate "
                "the measured scatter"
            ),
        },
    }


def _variability_assessment(exposures):
    accepted = [
        exposure
        for exposure in exposures
        if exposure["accepted_for_lya"]
        and exposure["red_wing"]["integrated_flux"] is not None
    ]
    if len(accepted) < 2:
        suspicious = any(
            exposure["automatic_extraction_suspicious"]
            for exposure in exposures
        )
        return {
            "red_wing_pairwise_comparisons": [],
            "largest_difference_significance_sigma": None,
            "single_shared_lya_amplitude_supported": None,
            "recommendation": (
                "Re-extract the flagged 2-D spectra before testing visit "
                "variability, coaddition, or a shared Ly-alpha amplitude."
                if suspicious
                else "Only one usable spectrum is present; no exposure-level "
                "variability or coaddition comparison is possible."
            ),
        }

    comparisons = []
    largest_significance = 0.0
    for first_index, first in enumerate(accepted):
        for second in accepted[first_index + 1 :]:
            first_red = first["red_wing"]
            second_red = second["red_wing"]
            combined_error = np.hypot(
                first_red["uncertainty"],
                second_red["uncertainty"],
            )
            significance = abs(
                first_red["integrated_flux"]
                - second_red["integrated_flux"]
            ) / combined_error
            largest_significance = max(largest_significance, significance)
            comparisons.append(
                {
                    "exposures": [first["rootname"], second["rootname"]],
                    "difference_significance_sigma": float(significance),
                }
            )
    shared_amplitude_supported = largest_significance < 5.0
    return {
        "red_wing_pairwise_comparisons": comparisons,
        "largest_difference_significance_sigma": float(
            largest_significance
        ),
        "single_shared_lya_amplitude_supported": shared_amplitude_supported,
        "recommendation": (
            "A shared Ly-alpha amplitude is not rejected by this diagnostic."
            if shared_amplitude_supported
            else "Do not coadd all visits or force one shared Ly-alpha "
            "amplitude; fit visits separately or add exposure/visit-specific "
            "amplitudes."
        ),
    }


def _plot_report(
    spectra,
    assessments,
    output_path,
    *,
    target_name,
    stellar_rv,
    fit_window,
    red_wing_window,
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    full_axis, lya_axis, flux_axis = axes

    for index, (spectrum, assessment) in enumerate(
        zip(spectra, assessments)
    ):
        color = colors[index % len(colors)]
        label = assessment["rootname"]
        accepted = assessment["accepted_for_lya"]
        style = "-" if accepted else "--"
        alpha = 0.9 if accepted else 0.45
        full_axis.plot(
            spectrum.wavelength,
            spectrum.flux,
            style,
            color=color,
            alpha=alpha,
            linewidth=0.8,
            label=label + ("" if accepted else " (rejected)"),
        )
        zoom = _window_mask(spectrum.wavelength, fit_window)
        lya_axis.step(
            spectrum.wavelength[zoom],
            spectrum.flux[zoom],
            where="mid",
            color=color,
            alpha=alpha,
            linewidth=1.0,
            label=label,
        )
        bad = zoom & ~spectrum.quality_mask
        if np.any(bad):
            lya_axis.scatter(
                spectrum.wavelength[bad],
                spectrum.flux[bad],
                marker="x",
                color=color,
                alpha=0.65,
                s=18,
            )
        airglow_window = assessment["airglow_exclusion_window_angstrom"]
        if airglow_window is not None and accepted:
            lya_axis.axvspan(
                *airglow_window,
                color=color,
                alpha=0.055,
                linewidth=0.0,
            )

    stellar_center = LYA_REST_WAVELENGTH * (
        1.0 + stellar_rv / SPEED_OF_LIGHT_KM_S
    ) if stellar_rv is not None else None
    if stellar_center is not None:
        lya_axis.axvline(
            stellar_center,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"stellar RV center ({stellar_rv:g} km/s)",
        )
    lya_axis.axvspan(
        *red_wing_window,
        color="gold",
        alpha=0.10,
        label="red-wing diagnostic",
    )

    names = [assessment["rootname"] for assessment in assessments]
    for index, assessment in enumerate(assessments):
        red = assessment["red_wing"]
        if red["integrated_flux"] is None:
            flux_axis.scatter(index, 0.0, marker="x", color="0.45")
            continue
        if not assessment["accepted_for_lya"]:
            flux_axis.scatter(
                index,
                red["integrated_flux"],
                marker="x",
                color="0.45",
                s=42,
            )
            continue
        flux_axis.errorbar(
            index,
            red["integrated_flux"],
            yerr=red["uncertainty"],
            marker="o",
            capsize=3,
            color=colors[index % len(colors)],
        )

    full_axis.set_xlim(1194.0, 1250.0)
    full_axis.set_ylabel("Flux density")
    full_axis.set_title(f"{target_name}: all supplied G140M exposures")
    full_axis.legend(fontsize=8, ncol=3)
    full_axis.axhline(0.0, color="0.35", linewidth=0.6)

    lya_axis.set_xlim(*fit_window)
    lya_axis.set_ylabel("Flux density")
    lya_axis.set_title(
        "Ly-alpha audit (x = pipeline-serious DQ; faint bands = airglow masks)"
    )
    lya_axis.legend(fontsize=8, ncol=3)
    lya_axis.axhline(0.0, color="0.35", linewidth=0.6)

    flux_axis.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    flux_axis.set_ylabel("Integrated red-wing flux")
    flux_axis.set_title(
        "Visit/exposure consistency diagnostic (x = invalid extraction)"
    )
    flux_axis.axhline(0.0, color="0.35", linewidth=0.6)
    flux_axis.set_xlabel("Exposure rootname")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_directory",
        nargs="?",
        type=Path,
        default=Path("user_data/stis/TOI178/G140M"),
    )
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--target-name", default="TOI-178")
    parser.add_argument(
        "--stellar-rv",
        type=float,
        default=57.4,
        help=(
            "heliocentric systemic velocity in km/s; the default is the "
            "Gaia DR2 value reported for TOI-178"
        ),
    )
    parser.add_argument(
        "--coadded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record whether each supplied spectrum is already a coadd",
    )
    parser.add_argument(
        "--fit-window",
        type=float,
        nargs=2,
        default=DEFAULT_FIT_WINDOW,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--red-wing-window",
        type=float,
        nargs=2,
        default=DEFAULT_RED_WING_WINDOW,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--airglow-half-width",
        type=float,
        default=0.12,
        help="half width of the conservative per-exposure mask in Angstrom",
    )
    parser.add_argument(
        "--maximum-extraction-offset",
        type=float,
        default=20.0,
        help=(
            "flag larger automatic cross-dispersion shifts for manual "
            "inspection; this is a workflow heuristic, not an STScI limit"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("user_data/validation/toi178-g140m"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    paths = sorted(arguments.input_directory.glob(arguments.pattern))
    if not paths:
        raise SystemExit(
            f"No files matching {arguments.pattern!r} in "
            f"{arguments.input_directory}"
        )

    spectra = []
    for path in paths:
        spectrum = load_stis_x1d(
            path,
            metadata_overrides={
                "target_name": arguments.target_name,
                "coadded": arguments.coadded,
            },
        )
        if spectrum.metadata["grating"] != "G140M":
            print(
                f"Skipping {path.name}: header grating is "
                f"{spectrum.metadata['grating']}, not G140M"
            )
            continue
        spectra.append(spectrum)
    if not spectra:
        raise SystemExit("No supplied FITS file is a G140M x1d product")

    assessments = [
        assess_exposure(
            spectrum,
            fit_window=tuple(arguments.fit_window),
            red_wing_window=tuple(arguments.red_wing_window),
            noise_windows=DEFAULT_NOISE_WINDOWS,
            airglow_half_width=arguments.airglow_half_width,
            maximum_extraction_offset=arguments.maximum_extraction_offset,
        )
        for spectrum in spectra
    ]
    variability = _variability_assessment(assessments)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = arguments.output_directory / "g140m_exposure_audit.png"
    report_path = arguments.output_directory / "g140m_exposure_audit.json"
    _plot_report(
        spectra,
        assessments,
        plot_path,
        target_name=arguments.target_name,
        stellar_rv=arguments.stellar_rv,
        fit_window=tuple(arguments.fit_window),
        red_wing_window=tuple(arguments.red_wing_window),
    )

    report = {
        "workflow": "real STIS G140M pre-fit validation",
        "target_name": arguments.target_name,
        "stellar_radial_velocity_km_s": arguments.stellar_rv,
        "stellar_radial_velocity_source": (
            "Gaia DR2 value cited by Leleu et al. 2021"
            if arguments.target_name == "TOI-178"
            and np.isclose(arguments.stellar_rv, 57.4)
            else "user-supplied CLI value"
        ),
        "input_directory": str(arguments.input_directory.resolve()),
        "fit_window_angstrom": list(arguments.fit_window),
        "dq_policy": (
            "Exclude bits in each SCI extension's SDQFLAGS; retain pixels on "
            "their native grids."
        ),
        "airglow_policy": (
            "Predict the terrestrial Ly-alpha position separately from each "
            "exposure's V_HELIO and define, but do not subtract, a mask."
        ),
        "extraction_policy": (
            "OFFSET records how far the automatic cross-correlation moved "
            "from its starting trace position. Large shifts are flagged for "
            "2-D inspection and fixed-position re-extraction before any "
            "one-dimensional scientific comparison."
        ),
        "exposures": assessments,
        "visit_variability": variability,
        "files": {"plot": str(plot_path.resolve())},
        "scope": (
            "This audit does not re-extract 2-D data or reconstruct intrinsic "
            "Ly-alpha. Fitted "
            "continuum/background/airglow components are a later workflow "
            "step."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path.resolve()}")
    print(f"Wrote {plot_path.resolve()}")
    print(variability["recommendation"])


if __name__ == "__main__":
    main()
