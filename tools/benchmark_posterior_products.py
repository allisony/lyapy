"""Benchmark reusable batched posterior reconstruction for GJ 411."""

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
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.composable import load_gj411_composable_problem
from lyapy.inference.reconstruction import (
    ReconstructionEvaluator,
    evaluate_reconstruction_samples,
)


def make_draws(configuration, number, seed):
    random = np.random.default_rng(seed)
    columns = []
    for parameter in configuration.free_parameters:
        baseline = configuration.base_parameters[parameter.index]
        width = parameter.upper - parameter.lower
        values = baseline + random.normal(scale=0.02 * width, size=number)
        columns.append(np.clip(values, parameter.lower, parameter.upper))
    return np.column_stack(columns)


def elapsed_seconds(function, repeats):
    start = perf_counter()
    result = None
    for _ in range(repeats):
        result = function()
    return (perf_counter() - start) / repeats, result


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("posterior_product_benchmark"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.draws < 1 or arguments.repeats < 1:
        raise SystemExit("--draws and --repeats must be positive")
    problem = load_gj411_composable_problem(oversample=arguments.oversample)
    draws = make_draws(
        problem.configuration,
        arguments.draws,
        arguments.seed,
    )

    reusable = ReconstructionEvaluator(
        problem.data,
        problem.configuration,
        forward_model=problem.forward_model,
    )
    reusable.evaluate(draws)
    evaluate_reconstruction_samples(
        problem.data,
        problem.configuration,
        draws,
        forward_model=problem.forward_model,
    )
    temporary_seconds, temporary = elapsed_seconds(
        lambda: evaluate_reconstruction_samples(
            problem.data,
            problem.configuration,
            draws,
            forward_model=problem.forward_model,
        ),
        arguments.repeats,
    )
    reusable_seconds, repeated = elapsed_seconds(
        lambda: reusable.evaluate(draws),
        arguments.repeats,
    )
    chunked = ReconstructionEvaluator(
        problem.data,
        problem.configuration,
        forward_model=problem.forward_model,
        batch_size=arguments.batch_size,
    ).evaluate(draws)

    maximum_difference = float(
        max(
            np.max(np.abs(temporary.intrinsic - repeated.intrinsic)),
            np.max(
                np.abs(
                    temporary.convolved_observed
                    - repeated.convolved_observed
                )
            ),
            np.max(np.abs(temporary.intrinsic - chunked.intrinsic)),
        )
    )
    report = {
        "problem": "GJ 411 composable posterior-product performance",
        "settings": {
            "draws": arguments.draws,
            "oversample": arguments.oversample,
            "repeats": arguments.repeats,
            "chunk_batch_size": arguments.batch_size,
            "seed": arguments.seed,
        },
        "temporary_evaluator_seconds": temporary_seconds,
        "reusable_evaluator_seconds": reusable_seconds,
        "temporary_evaluator_draws_per_second": (
            arguments.draws / temporary_seconds
        ),
        "reusable_evaluator_draws_per_second": (
            arguments.draws / reusable_seconds
        ),
        "reusable_speedup": temporary_seconds / reusable_seconds,
        "maximum_absolute_numerical_difference": maximum_difference,
        "chunked_output_matches": bool(maximum_difference < 1e-20),
    }

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    labels = ("temporary evaluator", "reusable evaluator")
    times = (temporary_seconds, reusable_seconds)
    bars = axis.bar(labels, times, color=("0.6", "tab:blue"))
    axis.bar_label(bars, fmt="%.3f s")
    axis.set_ylabel(f"seconds per {arguments.draws} posterior draws")
    axis.set_title(
        "Reusable compiled posterior reconstruction\n"
        f"speedup = {report['reusable_speedup']:.2f}x"
    )
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
