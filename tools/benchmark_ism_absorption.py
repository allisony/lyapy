"""Benchmark direct-index versus binary-search Harris interpolation."""

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

from lyapy.inference.composable import load_gj411_composable_problem
from lyapy.jax_models import (
    CGS_SPEED_OF_LIGHT,
    ELECTRON_CHARGE_ESU,
    ELECTRON_MASS_G,
    KM_SPEED_OF_LIGHT,
    _LEGACY_H1,
    _LEGACY_H3,
    _LEGACY_U_VALUES,
)


def binary_search_harris(a_parameter, u_parameter):
    """Pre-optimization Harris evaluation retained only as a benchmark."""
    u = jnp.abs(u_parameter)
    u_squared = u * u
    safe_u_squared = jnp.where(u_squared == 0.0, 1.0, u_squared)
    safe_u_fourth = safe_u_squared * safe_u_squared
    asymptotic = a_parameter / (jnp.sqrt(jnp.pi) * safe_u_squared) * (
        1.0
        + 1.5 / safe_u_squared
        + 3.75 / safe_u_fourth
        - a_parameter**2
        / safe_u_squared
        * (1.0 + 5.0 / safe_u_squared + 26.25 / safe_u_fourth)
    )
    exponential = jnp.exp(-u_squared)
    table_h1 = jnp.interp(u, _LEGACY_U_VALUES, _LEGACY_H1)
    table_h3 = jnp.interp(u, _LEGACY_U_VALUES, _LEGACY_H3)
    table = exponential + a_parameter * (
        table_h1
        + a_parameter
        * (
            (1.0 - 2.0 * u_squared) * exponential
            + a_parameter
            * (
                table_h3
                + a_parameter
                * (0.5 - 2.0 * u_squared + 2.0 / 3.0 * u_squared**2)
                * exponential
            )
        )
    )
    return jnp.where(u > 19.99, asymptotic, table)


def reference_optical_depth(
    wavelength,
    log_column,
    velocity,
    doppler_b,
    rest_wavelength,
    oscillator_strength,
    damping_rate,
):
    column_density = 10.0**log_column
    frequency_center = CGS_SPEED_OF_LIGHT / (rest_wavelength * 1.0e-8)
    doppler_frequency = frequency_center * doppler_b / KM_SPEED_OF_LIGHT
    a_parameter = jnp.abs(damping_rate / (4.0 * jnp.pi * doppler_frequency))
    cross_section_center = (
        jnp.sqrt(jnp.pi)
        * ELECTRON_CHARGE_ESU**2
        * oscillator_strength
        * rest_wavelength
        / (ELECTRON_MASS_G * CGS_SPEED_OF_LIGHT * doppler_b * 1.0e13)
    )
    rest_frame_frequency = (
        CGS_SPEED_OF_LIGHT
        / (wavelength * 1.0e-8)
        * (1.0 + velocity / KM_SPEED_OF_LIGHT)
    )
    u_parameter = (rest_frame_frequency - frequency_center) / doppler_frequency
    return (
        cross_section_center
        * binary_search_harris(a_parameter, u_parameter)
        * column_density
    )


def reference_transmission(wavelength, parameters):
    hydrogen_tau = reference_optical_depth(
        wavelength,
        parameters[4],
        parameters[6],
        parameters[5],
        1215.67,
        0.4161,
        6.26e8,
    )
    deuterium_tau = reference_optical_depth(
        wavelength,
        parameters[4] + jnp.log10(1.5e-5),
        parameters[6],
        parameters[5] / jnp.sqrt(2.0),
        1215.3394,
        0.4161,
        6.27e8,
    )
    return jnp.exp(-(hydrogen_tau + deuterium_tau))


def average_seconds(function, argument, repeats):
    value = function(argument)
    jax.block_until_ready(value)
    start = perf_counter()
    for _ in range(repeats):
        value = function(argument)
        jax.block_until_ready(value)
    return (perf_counter() - start) / repeats


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("ism_absorption_benchmark"),
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

    old_absorption = jax.jit(
        lambda values: reference_transmission(model_wavelength, values)
    )
    new_absorption = jax.jit(
        lambda values: model.absorption_model.evaluate(
            model_wavelength,
            values,
        ).transmission
    )

    def old_forward(values):
        intrinsic = model.emission_model.evaluate(
            model_wavelength,
            values,
        ).intrinsic
        _, detector_flux = operator.apply(
            intrinsic * reference_transmission(model_wavelength, values),
            values,
        )
        return detector_flux

    old_forward = jax.jit(old_forward)
    new_forward = jax.jit(
        lambda values: model(
            data_wavelength,
            values,
            data_kernel,
        ).convolved.observed
    )
    old_absorption_value = old_absorption(parameters)
    new_absorption_value = new_absorption(parameters)
    old_detector_value = old_forward(parameters)
    new_detector_value = new_forward(parameters)
    timings = {}
    for name, old_function, new_function in (
        ("absorption", old_absorption, new_absorption),
        ("complete_forward_model", old_forward, new_forward),
    ):
        old_seconds = average_seconds(
            old_function,
            parameters,
            arguments.repeats,
        )
        new_seconds = average_seconds(
            new_function,
            parameters,
            arguments.repeats,
        )
        timings[name] = {
            "binary_search_seconds": old_seconds,
            "direct_index_seconds": new_seconds,
            "speedup": old_seconds / new_seconds,
        }
    report = {
        "problem": "GJ 411 Harris-table interpolation performance",
        "settings": {
            "oversample": arguments.oversample,
            "high_resolution_cells": int(model_wavelength.size),
            "repeats": arguments.repeats,
        },
        "timings": timings,
        "maximum_absorption_difference": float(
            jnp.max(jnp.abs(old_absorption_value - new_absorption_value))
        ),
        "maximum_detector_flux_difference": float(
            jnp.max(jnp.abs(old_detector_value - new_detector_value))
        ),
    }

    labels = ("ISM absorption", "complete forward model")
    old_times = [
        1e3 * timings[name]["binary_search_seconds"]
        for name in ("absorption", "complete_forward_model")
    ]
    new_times = [
        1e3 * timings[name]["direct_index_seconds"]
        for name in ("absorption", "complete_forward_model")
    ]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(
        positions - width / 2,
        old_times,
        width,
        label="binary-search interpolation",
        color="0.6",
    )
    axis.bar(
        positions + width / 2,
        new_times,
        width,
        label="direct-index interpolation",
        color="tab:blue",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("milliseconds per evaluation")
    axis.set_title("Uniform-grid Harris interpolation performance")
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
