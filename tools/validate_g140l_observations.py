"""Validate real STIS G140L Ly-alpha exposures on their native grids.

The default dataset is the local HIP 67522 example. This workflow audits the
FITS headers, data-quality flags, extraction consistency, wavelength grids,
airglow location, exposure-to-exposure red-side Ly-alpha flux, and the bundled
G140L LSF. It does not re-extract spectra, coadd different grids, subtract
airglow, or perform a posterior fit.

Run from the repository root::

    python tools/validate_g140l_observations.py
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference import build_g140l_observation_operator  # noqa: E402
from lyapy.observations import (  # noqa: E402
    STIS_DQ_FLAGS,
    load_stis_lsf,
    load_stis_x1d,
    stis_airglow_wavelength,
)


DEFAULT_FIT_WINDOW = (1205.0, 1227.0)
# Si III 1206.5 is deliberately excluded. At G140L resolution, the red side
# is the cleanest simple Ly-alpha-only diagnostic available before the fitted
# continuum/background and airglow work in the next roadmap step.
DEFAULT_LYA_DIAGNOSTIC_WINDOWS = ((1217.0, 1222.0),)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _window_mask(wavelength, window):
    lower, upper = window
    return (wavelength >= lower) & (wavelength <= upper)


def _union_mask(wavelength, windows):
    mask = np.zeros(wavelength.shape, dtype=bool)
    for window in windows:
        mask |= _window_mask(wavelength, window)
    return mask


def _integrated_windows(spectrum, windows):
    mask = _union_mask(spectrum.wavelength, windows) & spectrum.quality_mask
    if np.count_nonzero(mask) < 2:
        return None, None, int(np.count_nonzero(mask))
    widths = np.gradient(spectrum.wavelength)
    flux = float(np.sum(spectrum.flux[mask] * widths[mask]))
    error = float(
        np.sqrt(np.sum((spectrum.error[mask] * widths[mask]) ** 2))
    )
    return flux, error, int(np.count_nonzero(mask))


def _extraction_consistency(spectra, tolerance_pixels):
    centers = np.asarray(
        [
            spectrum.metadata.get("extraction_center_a2_pixels")
            for spectrum in spectra
        ],
        dtype=np.float64,
    )
    offsets = np.asarray(
        [
            spectrum.metadata.get("extraction_offset_pixels")
            for spectrum in spectra
        ],
        dtype=np.float64,
    )
    complete = np.all(np.isfinite(centers)) and np.all(np.isfinite(offsets))
    center_span = float(np.ptp(centers)) if complete else None
    offset_span = float(np.ptp(offsets)) if complete else None
    consistent = bool(
        complete
        and center_span <= tolerance_pixels
        and offset_span <= tolerance_pixels
    )
    return {
        "consistent_across_exposures": consistent,
        "a2center_median_pixels": (
            float(np.median(centers)) if complete else None
        ),
        "a2center_peak_to_peak_pixels": center_span,
        "offset_median_pixels": (
            float(np.median(offsets)) if complete else None
        ),
        "offset_peak_to_peak_pixels": offset_span,
        "consistency_tolerance_pixels": tolerance_pixels,
        "interpretation": (
            "Consistency is a group diagnostic, not proof of a correct "
            "extraction. Lyapy requires users to supply scientifically "
            "reviewed one-dimensional spectra."
        ),
    }


def _grid_comparison(spectra):
    exact = True
    maximum_shift = 0.0
    for first_index, first in enumerate(spectra):
        for second in spectra[first_index + 1 :]:
            if not np.array_equal(first.wavelength, second.wavelength):
                exact = False
            if first.wavelength.shape == second.wavelength.shape:
                maximum_shift = max(
                    maximum_shift,
                    float(
                        np.max(
                            np.abs(first.wavelength - second.wavelength)
                        )
                    ),
                )
            else:
                maximum_shift = float("nan")
    dispersions = np.asarray(
        [np.median(np.diff(spectrum.wavelength)) for spectrum in spectra]
    )
    median_dispersion = float(np.median(dispersions))
    return {
        "exactly_identical": exact,
        "median_dispersion_angstrom_per_pixel": median_dispersion,
        "minimum_dispersion_angstrom_per_pixel": float(
            np.min(dispersions)
        ),
        "maximum_dispersion_angstrom_per_pixel": float(
            np.max(dispersions)
        ),
        "maximum_same-index_shift_angstrom": maximum_shift,
        "maximum_shift_detector_pixels": (
            None
            if not np.isfinite(maximum_shift)
            else maximum_shift / median_dispersion
        ),
        "recommendation": (
            "Retain native grids for joint inference; do not silently "
            "interpolate or use exact-grid inverse-variance coaddition."
            if not exact
            else "Exact-grid coaddition can be considered only after noise, "
            "mask, LSF, and variability assumptions are also verified."
        ),
    }


def _lsf_and_operator_summary(spectra, fit_window, oversample):
    lsf = load_stis_lsf(
        "G140L",
        spectra[0].metadata["aperture"],
        wavelength=1215.67,
    )
    normalized = lsf.profile / np.sum(lsf.profile)
    mean = float(np.sum(lsf.offset * normalized))
    sigma_pixels = float(
        np.sqrt(np.sum((lsf.offset - mean) ** 2 * normalized))
    )
    half_maximum = 0.5 * np.max(normalized)
    half_maximum_offsets = lsf.offset[normalized >= half_maximum]
    fwhm_pixels = float(
        half_maximum_offsets[-1] - half_maximum_offsets[0]
    )
    median_dispersion = float(
        np.median(
            [
                np.median(np.diff(spectrum.wavelength))
                for spectrum in spectra
            ]
        )
    )

    operator_summaries = []
    for spectrum in spectra:
        fit_mask = _window_mask(spectrum.wavelength, fit_window)
        wavelength = spectrum.wavelength[fit_mask]
        operator = build_g140l_observation_operator(
            wavelength,
            aperture=spectrum.metadata["aperture"],
            oversample=oversample,
            model_padding=4.0,
        )
        row_sums = np.sum(operator.binning_weights, axis=1)
        operator_summaries.append(
            {
                "rootname": spectrum.metadata["rootname"],
                "detector_bins": int(wavelength.size),
                "high_resolution_cells": int(operator.wavelength.size),
                "oversample": operator.oversample,
                "binning_weight_maximum_error": float(
                    np.max(np.abs(row_sums - 1.0))
                ),
                "convolution_kernel_sum": float(
                    np.sum(operator.convolution_kernel)
                ),
            }
        )
    return {
        "reference_wavelength_angstrom": 1200.0,
        "aperture": spectra[0].metadata["aperture"],
        "fwhm_detector_pixels": fwhm_pixels,
        "fwhm_angstrom_at_median_dispersion": (
            fwhm_pixels * median_dispersion
        ),
        "rms_width_detector_pixels": sigma_pixels,
        "rms_width_angstrom_at_median_dispersion": (
            sigma_pixels * median_dispersion
        ),
        "operator_checks": operator_summaries,
        "operation_order": (
            "oversampled physical model -> ISM absorption -> G140L LSF "
            "convolution -> integration over each native detector bin"
        ),
    }


def _exposure_assessment(
    spectrum,
    *,
    fit_window,
    diagnostic_windows,
    airglow_half_width,
    extraction_consistent,
):
    fit_mask = _window_mask(spectrum.wavelength, fit_window)
    usable = fit_mask & spectrum.quality_mask
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
    velocity = spectrum.metadata.get("heliocentric_velocity_km_s")
    airglow_center = (
        None if velocity is None else stis_airglow_wavelength(velocity)
    )
    diagnostic_flux, diagnostic_error, diagnostic_pixels = _integrated_windows(
        spectrum,
        diagnostic_windows,
    )
    accepted = np.count_nonzero(usable) >= 2 and extraction_consistent
    return {
        "filename": spectrum.metadata["filename"],
        "rootname": spectrum.metadata["rootname"],
        "observation_set": spectrum.metadata["observation_set"],
        "date_obs": spectrum.metadata["date_obs"],
        "time_obs": spectrum.metadata["time_obs"],
        "exposure_time_seconds": spectrum.metadata["exposure_time_seconds"],
        "accepted_for_validation": bool(accepted),
        "fit_pixels": int(np.count_nonzero(fit_mask)),
        "usable_fit_pixels": int(np.count_nonzero(usable)),
        "fit_window_dq_flags": flag_counts,
        "a2center_pixels": spectrum.metadata.get(
            "extraction_center_a2_pixels"
        ),
        "extraction_offset_pixels": spectrum.metadata.get(
            "extraction_offset_pixels"
        ),
        "extraction_max_search_pixels": spectrum.metadata.get(
            "extraction_max_search_pixels"
        ),
        "heliocentric_velocity_km_s": velocity,
        "predicted_airglow_center_angstrom": airglow_center,
        "airglow_diagnostic_window_angstrom": (
            None
            if airglow_center is None
            else [
                airglow_center - airglow_half_width,
                airglow_center + airglow_half_width,
            ]
        ),
        "red_side_lya_diagnostic": {
            "windows_angstrom": [
                list(window) for window in diagnostic_windows
            ],
            "integrated_flux": diagnostic_flux,
            "uncertainty": diagnostic_error,
            "signal_to_noise": (
                None
                if diagnostic_flux is None
                or diagnostic_error in (None, 0.0)
                else diagnostic_flux / diagnostic_error
            ),
            "pixels": diagnostic_pixels,
            "excluded_nearby_line": "Si III 1206.5 Angstrom",
            "interpretation": (
                "Observed red-side Ly-alpha-region flux only; not intrinsic "
                "or total reconstructed Ly-alpha flux."
            ),
        },
    }


def _variability_summary(assessments):
    accepted = [
        assessment
        for assessment in assessments
        if assessment["accepted_for_validation"]
        and assessment["red_side_lya_diagnostic"]["integrated_flux"] is not None
    ]
    comparisons = []
    maximum_significance = 0.0
    for first_index, first in enumerate(accepted):
        for second in accepted[first_index + 1 :]:
            first_flux = first["red_side_lya_diagnostic"]["integrated_flux"]
            second_flux = second["red_side_lya_diagnostic"]["integrated_flux"]
            uncertainty = np.hypot(
                first["red_side_lya_diagnostic"]["uncertainty"],
                second["red_side_lya_diagnostic"]["uncertainty"],
            )
            significance = abs(first_flux - second_flux) / uncertainty
            maximum_significance = max(maximum_significance, significance)
            comparisons.append(
                {
                    "rootnames": [first["rootname"], second["rootname"]],
                    "difference_significance_sigma": float(significance),
                }
            )

    visit_summaries = {}
    for visit in sorted(
        {assessment["observation_set"] for assessment in accepted}
    ):
        visit_exposures = [
            assessment
            for assessment in accepted
            if assessment["observation_set"] == visit
        ]
        flux = np.asarray(
            [
                exposure["red_side_lya_diagnostic"]["integrated_flux"]
                for exposure in visit_exposures
            ]
        )
        error = np.asarray(
            [
                exposure["red_side_lya_diagnostic"]["uncertainty"]
                for exposure in visit_exposures
            ]
        )
        weights = 1.0 / error**2
        visit_summaries[str(visit)] = {
            "rootnames": [
                exposure["rootname"] for exposure in visit_exposures
            ],
            "inverse_variance_mean_red_side_flux": float(
                np.sum(weights * flux) / np.sum(weights)
            ),
            "statistical_uncertainty": float(1.0 / np.sqrt(np.sum(weights))),
        }

    visit_difference_significance = None
    if len(visit_summaries) == 2:
        first_visit, second_visit = visit_summaries.values()
        visit_difference_significance = abs(
            first_visit["inverse_variance_mean_red_side_flux"]
            - second_visit["inverse_variance_mean_red_side_flux"]
        ) / np.hypot(
            first_visit["statistical_uncertainty"],
            second_visit["statistical_uncertainty"],
        )

    shared_amplitude = maximum_significance < 5.0
    return {
        "pairwise_comparisons": comparisons,
        "maximum_pairwise_difference_sigma": float(maximum_significance),
        "visit_summaries": visit_summaries,
        "visit_mean_difference_sigma": (
            None
            if visit_difference_significance is None
            else float(visit_difference_significance)
        ),
        "single_shared_amplitude_supported": shared_amplitude,
        "recommendation": (
            "A single shared amplitude is not rejected by this red-side "
            "Ly-alpha-region diagnostic."
            if shared_amplitude
            else "Individual exposures reject one perfectly shared Ly-alpha "
            "amplitude at the reported statistical-error level. Consider "
            "exposure-specific amplitudes or an extra-scatter model; inspect "
            "the visit means before assigning visit-level variability."
        ),
        "caveat": (
            "This is a raw red-side flux diagnostic, not an astrophysical "
            "variability claim; background and systematic uncertainty are "
            "addressed in the next roadmap step."
        ),
    }


def _plot(spectra, assessments, output_path, fit_window, diagnostic_windows):
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    full_axis, lya_axis, flux_axis = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (spectrum, assessment) in enumerate(
        zip(spectra, assessments)
    ):
        color = colors[index % len(colors)]
        label = assessment["rootname"]
        full_axis.plot(
            spectrum.wavelength,
            spectrum.flux,
            color=color,
            linewidth=0.75,
            label=label,
        )
        zoom = _window_mask(spectrum.wavelength, fit_window)
        lya_axis.step(
            spectrum.wavelength[zoom],
            spectrum.flux[zoom],
            where="mid",
            color=color,
            linewidth=1.0,
            label=label,
        )
        airglow_window = assessment[
            "airglow_diagnostic_window_angstrom"
        ]
        if airglow_window is not None:
            lya_axis.axvspan(
                *airglow_window,
                color=color,
                alpha=0.035,
                linewidth=0.0,
            )
        diagnostic = assessment["red_side_lya_diagnostic"]
        flux_axis.errorbar(
            index,
            diagnostic["integrated_flux"],
            yerr=diagnostic["uncertainty"],
            marker="o",
            capsize=3,
            color=color,
        )

    for window in diagnostic_windows:
        lya_axis.axvspan(*window, color="gold", alpha=0.065, linewidth=0.0)
    full_axis.set_xlim(1120.0, 1716.0)
    displayed_flux = np.concatenate(
        [
            spectrum.flux[
                (spectrum.wavelength >= 1180.0)
                & (spectrum.wavelength <= 1716.0)
            ]
            for spectrum in spectra
        ]
    )
    lower = float(np.percentile(displayed_flux, 0.3))
    upper = float(np.max(displayed_flux))
    span = upper - lower
    full_axis.set_ylim(lower - 0.05 * span, upper + 0.12 * span)
    full_axis.set_ylabel("Flux density")
    full_axis.set_title("HIP 67522: six native-grid G140L exposures")
    full_axis.legend(fontsize=8, ncol=3)
    full_axis.axhline(0.0, color="0.35", linewidth=0.6)

    lya_axis.set_xlim(*fit_window)
    lya_axis.set_ylabel("Flux density")
    lya_axis.set_title(
        "Ly-alpha sampling (faint bands = heuristic airglow regions)"
    )
    lya_axis.legend(fontsize=8, ncol=3)
    lya_axis.axhline(0.0, color="0.35", linewidth=0.6)

    names = [assessment["rootname"] for assessment in assessments]
    flux_axis.set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    flux_axis.set_ylabel("Integrated red-side flux")
    flux_axis.set_title(
        "Raw red-side Ly-alpha diagnostic (gold region in middle panel)"
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
        default=Path("user_data/stis/hip67522/g140l"),
    )
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--target-name", default="HIP 67522")
    parser.add_argument(
        "--fit-window",
        type=float,
        nargs=2,
        default=DEFAULT_FIT_WINDOW,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--airglow-half-width",
        type=float,
        default=1.2,
        help="diagnostic half width in Angstrom; no subtraction is performed",
    )
    parser.add_argument(
        "--extraction-consistency-tolerance",
        type=float,
        default=5.0,
        help="maximum group peak-to-peak A2CENTER and OFFSET spread in pixels",
    )
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("user_data/validation/hip67522-g140l"),
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
    skipped = []
    for path in paths:
        spectrum = load_stis_x1d(
            path,
            metadata_overrides={
                "target_name": arguments.target_name,
                "coadded": False,
            },
        )
        if spectrum.metadata["grating"] != "G140L":
            skipped.append(
                {
                    "filename": path.name,
                    "grating": spectrum.metadata["grating"],
                    "reason": "header grating is not G140L",
                }
            )
            continue
        spectra.append(spectrum)
    if len(spectra) < 2:
        raise SystemExit("G140L validation requires at least two exposures")

    extraction = _extraction_consistency(
        spectra,
        arguments.extraction_consistency_tolerance,
    )
    assessments = [
        _exposure_assessment(
            spectrum,
            fit_window=tuple(arguments.fit_window),
            diagnostic_windows=DEFAULT_LYA_DIAGNOSTIC_WINDOWS,
            airglow_half_width=arguments.airglow_half_width,
            extraction_consistent=extraction[
                "consistent_across_exposures"
            ],
        )
        for spectrum in spectra
    ]
    grids = _grid_comparison(spectra)
    lsf = _lsf_and_operator_summary(
        spectra,
        tuple(arguments.fit_window),
        arguments.oversample,
    )
    variability = _variability_summary(assessments)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = arguments.output_directory / "g140l_exposure_audit.png"
    report_path = arguments.output_directory / "g140l_exposure_audit.json"
    _plot(
        spectra,
        assessments,
        plot_path,
        tuple(arguments.fit_window),
        DEFAULT_LYA_DIAGNOSTIC_WINDOWS,
    )
    report = {
        "workflow": "real STIS G140L pre-fit validation",
        "target_name": arguments.target_name,
        "input_directory": str(arguments.input_directory.resolve()),
        "fit_window_angstrom": list(arguments.fit_window),
        "lya_diagnostic_windows_angstrom": [
            list(window) for window in DEFAULT_LYA_DIAGNOSTIC_WINDOWS
        ],
        "exposures": assessments,
        "skipped_files": skipped,
        "extraction_consistency": extraction,
        "native_wavelength_grids": grids,
        "g140l_lsf_and_observation_operator": lsf,
        "airglow_diagnostic": {
            "terrestrial_rest_wavelength_angstrom": 1215.67,
            "center_method": (
                "For a CALSTIS heliocentric wavelength grid, lambda_airglow "
                "= 1215.67 * (1 - V_HELIO / c)."
            ),
            "minimum_center_angstrom": float(
                min(
                    assessment["predicted_airglow_center_angstrom"]
                    for assessment in assessments
                )
            ),
            "maximum_center_angstrom": float(
                max(
                    assessment["predicted_airglow_center_angstrom"]
                    for assessment in assessments
                )
            ),
            "heuristic_half_width_angstrom": arguments.airglow_half_width,
            "half_width_in_point_source_lsf_fwhm": (
                arguments.airglow_half_width
                / lsf["fwhm_angstrom_at_median_dispersion"]
            ),
            "width_method": (
                "The default 1.2 Angstrom half-width is a conservative "
                "visual diagnostic based on the approximately 1.06 Angstrom "
                "bundled point-source G140L LSF FWHM. It was not measured "
                "from these spectra and is not a fitted airglow model."
            ),
            "caveat": (
                "Slit-filling geocoronal emission need not have exactly the "
                "point-source LSF. The next roadmap step will replace this "
                "display mask with explicit airglow treatment."
            ),
        },
        "exposure_variability": variability,
        "files": {"plot": str(plot_path.resolve())},
        "scope": (
            "This validates ingestion, sampling, masks, native grids, and "
            "the observation operator. Continuum/background and airglow "
            "models are intentionally deferred to the next roadmap step."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path.resolve()}")
    print(f"Wrote {plot_path.resolve()}")
    print(grids["recommendation"])
    print(variability["recommendation"])


if __name__ == "__main__":
    main()
