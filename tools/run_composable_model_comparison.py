"""Compare composable self-reversed and non-reversed models for GJ 411."""

import argparse
import json
from pathlib import Path
import sys

try:
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. From the repository root, run:\n\n"
        "    python -m pip install -e '.[inference]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import make_reversal_comparison_configurations
from lyapy.inference.comparison import (
    ModelHypothesis,
    compare_evidence,
    run_evidence_comparison,
)
from lyapy.inference.composable import load_gj411_composable_problem
from lyapy.inference.diagnostics import summarize_dynesty


def posterior_mean(sampler):
    """Return the dynesty-weighted mean free-parameter vector."""
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    weights /= np.sum(weights)
    return np.average(sampler.results.samples, axis=0, weights=weights)


def predict(problem, configuration, free_values):
    parameters = configuration.assemble_jax(free_values)
    return np.asarray(
        problem.forward_model(
            problem.data.wavelength,
            parameters,
            problem.data.convolution_kernel,
        ).convolved.observed
    )


def fit_summary(problem, configuration, free_values):
    model_flux = predict(problem, configuration, free_values)
    residuals = (problem.data.flux - model_flux) / problem.data.error
    degrees_of_freedom = problem.data.wavelength.size - configuration.ndim
    chi_squared = np.sum(residuals**2)
    return {
        "chi_squared": float(chi_squared),
        "degrees_of_freedom": int(degrees_of_freedom),
        "reduced_chi_squared": float(chi_squared / degrees_of_freedom),
        "residual_root_mean_square": float(np.sqrt(np.mean(residuals**2))),
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
    }


def make_plot(
    problem,
    self_reversed_configuration,
    no_reversal_configuration,
    self_reversed_sampler,
    no_reversal_sampler,
    comparison,
):
    self_reversed_model = predict(
        problem,
        self_reversed_configuration,
        posterior_mean(self_reversed_sampler),
    )
    no_reversal_model = predict(
        problem,
        no_reversal_configuration,
        posterior_mean(no_reversal_sampler),
    )
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    scale = 1.0e12
    profile_axis.errorbar(
        problem.data.wavelength,
        problem.data.flux * scale,
        yerr=problem.data.error * scale,
        color="0.65",
        linewidth=0.8,
        label="GJ 411 STIS data",
    )
    profile_axis.plot(
        problem.data.wavelength,
        self_reversed_model * scale,
        color="#d6278b",
        linewidth=2.0,
        label="self reversal",
    )
    profile_axis.plot(
        problem.data.wavelength,
        no_reversal_model * scale,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="no reversal",
    )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False)
    profile_axis.grid(alpha=0.2)

    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(
        problem.data.wavelength,
        (problem.data.flux - self_reversed_model) / problem.data.error,
        color="#d6278b",
    )
    residual_axis.plot(
        problem.data.wavelength,
        (problem.data.flux - no_reversal_model) / problem.data.error,
        color="#1f77b4",
        linestyle="--",
    )
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / error")
    residual_axis.grid(alpha=0.2)
    figure.suptitle(
        "Composable GJ 411 comparison: "
        f"Δlog Z = {comparison.delta_log_evidence:.1f} ± "
        f"{comparison.delta_log_evidence_error:.1f}"
    )
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--model-padding", type=float, default=1.0)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--dlogz", type=float, default=0.05)
    parser.add_argument(
        "--dynesty-sample",
        choices=("rwalk", "rslice"),
        default="rwalk",
    )
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--reversal-max", type=float, default=1.8)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("gj411_composable_model_comparison"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    problem = load_gj411_composable_problem(
        oversample=arguments.oversample,
        model_padding=arguments.model_padding,
    )
    self_reversed, no_reversal = make_reversal_comparison_configurations(
        problem.configuration,
        reversal_lower=0.0,
        reversal_upper=arguments.reversal_max,
    )
    selection = run_evidence_comparison(
        problem.data,
        (
            ModelHypothesis(
                "self reversal",
                self_reversed,
                forward_model=problem.forward_model,
            ),
            ModelHypothesis(
                "no reversal",
                no_reversal,
                forward_model=problem.forward_model,
            ),
        ),
        random_seed=arguments.seed,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        print_progress=not arguments.no_progress,
        sample=arguments.dynesty_sample,
        bootstrap=0,
    )
    self_reversed_run = selection.get_run("self reversal")
    no_reversal_run = selection.get_run("no reversal")
    comparison = compare_evidence(
        self_reversed_run.evidence,
        no_reversal_run.evidence,
    )
    self_reversed_names = tuple(
        parameter.name for parameter in self_reversed.free_parameters
    )
    no_reversal_names = tuple(
        parameter.name for parameter in no_reversal.free_parameters
    )
    self_reversed_mean = posterior_mean(self_reversed_run.sampler)
    no_reversal_mean = posterior_mean(no_reversal_run.sampler)
    report = {
        "problem": "observed GJ 411 composable self-reversal comparison",
        "seed": arguments.seed,
        "observation_operator": {
            "oversample": arguments.oversample,
            "model_padding_angstrom": arguments.model_padding,
            "order": "intrinsic emission * ISM, LSF convolution, detector integration",
        },
        "settings": {
            "live_points": arguments.nlive,
            "stopping_log_evidence": arguments.dlogz,
            "dynesty_sample": arguments.dynesty_sample,
        },
        "model_selection": selection.as_dict(),
        "comparison": comparison.as_dict(),
        "self_reversal": {
            "posterior": summarize_dynesty(
                self_reversed_run.sampler,
                self_reversed_names,
            ),
            "fit": fit_summary(problem, self_reversed, self_reversed_mean),
        },
        "no_reversal": {
            "posterior": summarize_dynesty(
                no_reversal_run.sampler,
                no_reversal_names,
            ),
            "fit": fit_summary(problem, no_reversal, no_reversal_mean),
        },
    }
    figure = make_plot(
        problem,
        self_reversed,
        no_reversal,
        self_reversed_run.sampler,
        no_reversal_run.sampler,
        comparison,
    )
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["comparison"], indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
