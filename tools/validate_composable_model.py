"""Validate high-resolution convolution, detector binning, and JAX performance."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

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

from lyapy.jax_components import CompositeEmissionModel, GaussianEmissionComponent
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator


def make_problem(oversample):
    detector_wavelength = np.linspace(1214.5, 1216.5, 101)
    lsf_offsets = np.linspace(-0.15, 0.15, 301)
    lsf_profile = np.exp(-0.5 * (lsf_offsets / 0.025) ** 2)
    operator = build_high_resolution_operator(
        detector_wavelength,
        lsf_offsets,
        lsf_profile,
        oversample=oversample,
        model_padding=0.5,
    )
    emission = CompositeEmissionModel(
        components=(GaussianEmissionComponent(1215.67, 0, 2),),
        log_total_flux_index=1,
    )
    model = ComposableBinnedModel(emission, operator)
    parameters = jnp.array([-15.0, -11.0, 5.0])
    return detector_wavelength, lsf_offsets, lsf_profile, operator, model, parameters


def bin_then_convolve(detector_wavelength, operator, intrinsic_high, offsets, lsf):
    """Deliberately coarse ordering used only to quantify its numerical bias."""
    binned_first = np.asarray(operator.binning_matrix) @ np.asarray(intrinsic_high)
    spacing = detector_wavelength[1] - detector_wavelength[0]
    radius = int(np.ceil(max(abs(offsets)) / spacing))
    coarse_offsets = spacing * np.arange(-radius, radius + 1)
    coarse_kernel = np.interp(coarse_offsets, offsets, lsf, left=0.0, right=0.0)
    coarse_kernel /= np.sum(coarse_kernel)
    return np.convolve(binned_first, coarse_kernel, mode="same")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("composable_validation.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON destination (defaults to the plot name with a .json suffix)",
    )
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.repeats < 1:
        raise SystemExit("--repeats must be positive")
    oversampling_values = (2, 4, 8, 16, 32)
    reference_oversample = 64
    predictions = {}
    for oversample in (*oversampling_values, reference_oversample):
        detector_wave, offsets, lsf, operator, model, parameters = make_problem(
            oversample
        )
        predictions[oversample] = np.asarray(
            model(detector_wave, parameters, np.array([1.0])).convolved.observed
        )
    reference = predictions[reference_oversample]
    relative_errors = {
        str(oversample): float(
            np.max(np.abs(predictions[oversample] - reference))
            / np.max(reference)
        )
        for oversample in oversampling_values
    }

    detector_wave, offsets, lsf, operator, model, parameters = make_problem(16)
    result = model(detector_wave, parameters, np.array([1.0]))
    wrong_order = bin_then_convolve(
        detector_wave,
        operator,
        result.unconvolved.intrinsic,
        offsets,
        lsf,
    )
    order_bias = float(
        np.max(np.abs(result.convolved.observed - wrong_order))
        / np.max(result.convolved.observed)
    )

    compiled = jax.jit(
        lambda values: model(
            detector_wave,
            values,
            np.array([1.0]),
        ).convolved.observed
    )
    start = perf_counter()
    compiled(parameters).block_until_ready()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(arguments.repeats):
        compiled(parameters).block_until_ready()
    evaluation_seconds = (perf_counter() - start) / arguments.repeats

    detector_widths = np.diff(operator.detector_edges)
    detector_integrated_flux = float(
        np.sum(np.asarray(result.convolved.observed) * detector_widths)
    )
    report = {
        "problem": "composable high-resolution observation operator validation",
        "sampled_total_intrinsic_flux": float(10.0 ** parameters[1]),
        "detector_window_integrated_flux": detector_integrated_flux,
        "reference_oversample": reference_oversample,
        "maximum_relative_error_vs_reference": relative_errors,
        "bin_then_convolve_relative_bias": order_bias,
        "jax_first_compile_and_evaluate_seconds": compile_seconds,
        "jax_compiled_evaluation_seconds": evaluation_seconds,
        "timing_repeats": arguments.repeats,
    }

    figure, (profile_axis, order_axis, convergence_axis) = plt.subplots(
        3,
        1,
        figsize=(9.5, 9.0),
        gridspec_kw={"height_ratios": (3, 1, 1)},
    )
    scale = 1e12
    profile_axis.plot(
        detector_wave,
        reference * scale,
        color="black",
        linewidth=2.5,
        label=f"high-resolution reference ({reference_oversample}×)",
    )
    profile_axis.plot(
        detector_wave,
        np.asarray(result.convolved.observed) * scale,
        color="#d6278b",
        linestyle="--",
        linewidth=2.0,
        label="convolve then bin (16×)",
    )
    profile_axis.plot(
        detector_wave,
        wrong_order * scale,
        color="#1f77b4",
        linestyle=":",
        linewidth=2.0,
        label="bin then coarse convolution",
    )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False)
    profile_axis.grid(alpha=0.2)

    peak = np.max(reference)
    order_axis.axhline(0.0, color="0.4", linewidth=1.0)
    order_axis.plot(
        detector_wave,
        100.0 * (np.asarray(result.convolved.observed) - reference) / peak,
        color="#d6278b",
        linestyle="--",
        label="correct 16× minus reference",
    )
    order_axis.plot(
        detector_wave,
        100.0 * (wrong_order - reference) / peak,
        color="#1f77b4",
        linestyle=":",
        label="wrong order minus reference",
    )
    order_axis.set_xlabel(r"wavelength ($\AA$)")
    order_axis.set_ylabel("peak-relative difference (%)")
    order_axis.legend(frameon=False, fontsize="small")
    order_axis.grid(alpha=0.2)

    convergence_axis.loglog(
        oversampling_values,
        [relative_errors[str(value)] for value in oversampling_values],
        marker="o",
        color="#d6278b",
    )
    convergence_axis.set_xlabel("internal samples per detector bin")
    convergence_axis.set_ylabel("maximum relative error")
    convergence_axis.grid(alpha=0.2, which="both")
    figure.suptitle("Composable forward-model numerical validation")
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
