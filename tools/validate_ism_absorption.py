"""Validate untruncated ISM damping wings and detector-grid convergence."""

import argparse
import json
from pathlib import Path
import sys

try:
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing JAX. From the repository root, run:\n\n"
        "    python -m pip install -e '.[jax]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.absorption import total_tau_profile
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    HydrogenDeuteriumAbsorber,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.jax_models import hydrogen_deuterium_transmission_jax


LINE_CENTER = 1215.67


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ism_absorption_validation.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON destination (defaults to the plot name with a .json suffix)",
    )
    return parser.parse_args()


def make_model(oversample):
    detector_wavelength = np.linspace(1214.5, 1216.55, 160)
    dispersion = np.median(np.diff(detector_wavelength))
    table = np.loadtxt(
        REPOSITORY_ROOT / "src" / "lyapy" / "STIS_E140M_LSF.dat",
        skiprows=2,
    )
    lsf_offsets = table[:, 0] * dispersion
    lsf_profile = table[:, 3]
    operator = build_high_resolution_operator(
        detector_wavelength,
        lsf_offsets,
        lsf_profile,
        oversample=oversample,
        model_padding=1.0,
    )
    emission = CompositeEmissionModel(
        components=(VoigtEmissionComponent(LINE_CENTER, 0, 3, 2),),
        log_total_flux_index=1,
    )
    absorption = CompositeAbsorptionModel(
        components=(HydrogenDeuteriumAbsorber(4, 5, 6),),
    )
    model = ComposableBinnedModel(
        emission,
        operator,
        absorption_model=absorption,
    )
    parameters = jnp.array([-15.0, -11.0, 9.0, 85.0, 18.9, 8.6, 3.0])
    return detector_wavelength, model, parameters


def main():
    arguments = parse_arguments()
    relative_wavelength = np.linspace(-20.0, 20.0, 80_001)
    wavelength = LINE_CENTER + relative_wavelength
    columns = (17.5, 18.0, 18.5, 18.9)
    transmissions = {
        column: np.asarray(
            hydrogen_deuterium_transmission_jax(
                wavelength,
                column,
                12.0,
                0.0,
            )
        )
        for column in columns
    }
    wing_diagnostics = {}
    for column, transmission in transmissions.items():
        absorption = 1.0 - transmission
        wing_diagnostics[str(column)] = {
            "absorption_at_two_angstroms": float(
                np.interp(2.0, relative_wavelength, absorption)
            ),
            "maximum_offset_above_one_percent_angstrom": float(
                np.max(np.abs(relative_wavelength[absorption > 0.01]))
            ),
            "maximum_offset_above_point_one_percent_angstrom": float(
                np.max(np.abs(relative_wavelength[absorption > 0.001]))
            ),
        }

    comparison_relative = np.linspace(-6.0, 6.0, 24_001)
    comparison_wavelength = LINE_CENTER + comparison_relative
    corrected = np.asarray(
        hydrogen_deuterium_transmission_jax(
            comparison_wavelength,
            18.9,
            12.0,
            0.0,
        )
    )
    legacy = total_tau_profile(
        comparison_wavelength,
        18.9,
        12.0,
        0.0,
        wave_cut_off=2.0,
    )

    oversampling_values = (2, 4, 8, 16, 32)
    reference_oversample = 64
    predictions = {}
    for oversample in (*oversampling_values, reference_oversample):
        detector_wavelength, model, parameters = make_model(oversample)
        predictions[oversample] = np.asarray(
            model(
                detector_wavelength,
                parameters,
                np.array([1.0]),
            ).convolved.observed
        )
    reference = predictions[reference_oversample]
    convergence = {
        str(oversample): float(
            np.max(np.abs(predictions[oversample] - reference))
            / np.max(reference)
        )
        for oversample in oversampling_values
    }

    report = {
        "problem": "untruncated ISM absorption and sampling validation",
        "doppler_b_km_s": 12.0,
        "wing_diagnostics": wing_diagnostics,
        "reference_oversample": reference_oversample,
        "maximum_detector_prediction_error_vs_reference": convergence,
    }

    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10.0))
    wing_axis, comparison_axis, convergence_axis = axes
    for column, transmission in transmissions.items():
        wing_axis.plot(
            relative_wavelength,
            transmission,
            label=rf"$\log N_{{\rm HI}}={column}$",
        )
    wing_axis.set_xlim(-10.0, 10.0)
    wing_axis.set_ylabel("ISM transmission")
    wing_axis.legend(frameon=False, ncol=2)
    wing_axis.grid(alpha=0.2)

    comparison_axis.plot(
        comparison_relative,
        corrected,
        color="black",
        linewidth=2.0,
        label="untruncated Voigt wings",
    )
    comparison_axis.plot(
        comparison_relative,
        legacy,
        color="#d6278b",
        linestyle="--",
        linewidth=2.0,
        label="legacy ±2 Å plateau",
    )
    comparison_axis.axvline(-2.0, color="0.5", linestyle=":", linewidth=1.0)
    comparison_axis.axvline(2.0, color="0.5", linestyle=":", linewidth=1.0)
    comparison_axis.set_ylabel("ISM transmission")
    comparison_axis.set_xlabel(r"offset from H I line center ($\AA$)")
    comparison_axis.legend(frameon=False)
    comparison_axis.grid(alpha=0.2)

    convergence_axis.loglog(
        oversampling_values,
        [convergence[str(value)] for value in oversampling_values],
        marker="o",
        color="#1f77b4",
    )
    convergence_axis.set_xlabel("internal samples per detector bin")
    convergence_axis.set_ylabel("maximum detector-space relative error")
    convergence_axis.grid(alpha=0.2, which="both")
    figure.suptitle("ISM Voigt-wing and sampling validation")
    figure.tight_layout()

    report_path = arguments.report or arguments.output.with_suffix(".json")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {arguments.output.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
