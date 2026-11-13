"""Benchmark LSF convolution and indexed detector-bin integration."""

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
from jax.scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.composable import load_gj411_composable_problem


def average_seconds(function, argument, repeats):
    value = function(argument)
    jax.block_until_ready(value)
    start = perf_counter()
    for _ in range(repeats):
        value = function(argument)
        jax.block_until_ready(value)
    return (perf_counter() - start) / repeats, value


def comparison(old_function, new_function, argument, repeats):
    old_seconds, old_value = average_seconds(
        old_function,
        argument,
        repeats,
    )
    new_seconds, new_value = average_seconds(
        new_function,
        argument,
        repeats,
    )
    return {
        "reference_seconds": old_seconds,
        "optimized_seconds": new_seconds,
        "speedup": old_seconds / new_seconds,
        "maximum_absolute_difference": float(
            jnp.max(jnp.abs(old_value - new_value))
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("instrument_operator_benchmark"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.repeats < 1:
        raise SystemExit("--repeats must be positive")
    problem = load_gj411_composable_problem(oversample=arguments.oversample)
    model = problem.forward_model
    operator = model.observation_operator
    parameters = jnp.asarray(problem.configuration.base_parameters)
    model_wavelength = jnp.asarray(operator.wavelength)
    data_wavelength = jnp.asarray(problem.data.wavelength)
    data_kernel = jnp.asarray(problem.data.convolution_kernel)
    kernel = jnp.asarray(operator.convolution_kernel)
    dense_matrix = jnp.asarray(operator.binning_matrix)
    indices = jnp.asarray(operator.binning_indices)
    weights = jnp.asarray(operator.binning_weights)
    intrinsic = model.emission_model.evaluate(
        model_wavelength,
        parameters,
    ).intrinsic
    transmission = model.absorption_model.evaluate(
        model_wavelength,
        parameters,
    ).transmission
    absorbed = intrinsic * transmission

    direct_convolution = jax.jit(
        lambda flux: jnp.convolve(flux, kernel, mode="same")
    )
    fft_convolution = jax.jit(
        lambda flux: fftconvolve(flux, kernel, mode="same")
    )
    convolved = direct_convolution(absorbed)
    dense_binning = jax.jit(lambda flux: dense_matrix @ flux)
    indexed_binning = jax.jit(
        lambda flux: jnp.sum(flux[indices] * weights, axis=1)
    )

    def dense_forward(values):
        emission = model.emission_model.evaluate(
            model_wavelength,
            values,
        ).intrinsic
        absorption = model.absorption_model.evaluate(
            model_wavelength,
            values,
        ).transmission
        blurred = jnp.convolve(emission * absorption, kernel, mode="same")
        return dense_matrix @ blurred

    dense_forward = jax.jit(dense_forward)
    indexed_forward = jax.jit(
        lambda values: model(
            data_wavelength,
            values,
            data_kernel,
        ).convolved.observed
    )

    convolution = comparison(
        direct_convolution,
        fft_convolution,
        absorbed,
        arguments.repeats,
    )
    binning = comparison(
        dense_binning,
        indexed_binning,
        convolved,
        arguments.repeats,
    )
    complete = comparison(
        dense_forward,
        indexed_forward,
        parameters,
        arguments.repeats,
    )
    dense_bytes = int(operator.binning_matrix.nbytes)
    indexed_bytes = int(
        operator.binning_indices.nbytes + operator.binning_weights.nbytes
    )
    report = {
        "problem": "GJ 411 high-resolution instrument-operator performance",
        "settings": {
            "oversample": arguments.oversample,
            "high_resolution_cells": int(model_wavelength.size),
            "lsf_kernel_samples": int(kernel.size),
            "detector_bins": int(problem.data.wavelength.size),
            "repeats": arguments.repeats,
        },
        "convolution_direct_reference_vs_fft_candidate": convolution,
        "detector_binning_dense_reference_vs_indexed": binning,
        "complete_forward_dense_reference_vs_indexed": complete,
        "binning_storage": {
            "dense_bytes": dense_bytes,
            "indexed_bytes": indexed_bytes,
            "reduction_factor": dense_bytes / indexed_bytes,
            "maximum_cells_per_detector_bin": int(indices.shape[1]),
        },
        "decision": (
            "retain direct convolution; use indexed detector-bin overlaps"
        ),
    }

    labels = ("detector binning", "complete forward model")
    references = (
        1e3 * binning["reference_seconds"],
        1e3 * complete["reference_seconds"],
    )
    optimized = (
        1e3 * binning["optimized_seconds"],
        1e3 * complete["optimized_seconds"],
    )
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(
        positions - width / 2,
        references,
        width,
        label="dense binning",
        color="0.6",
    )
    axis.bar(
        positions + width / 2,
        optimized,
        width,
        label="indexed binning",
        color="tab:blue",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("milliseconds per evaluation")
    axis.set_title("Sparse detector-bin overlap performance")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
