"""Compare reported, rescaled, jittered, and robust GJ 411 uncertainties."""

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

from lyapy.inference.comparison import ModelHypothesis, run_evidence_comparison
from lyapy.inference.composable import load_gj411_composable_problem
from lyapy.inference.diagnostics import summarize_dynesty
from lyapy.inference.model_catalog import build_gj411_model_catalog
from lyapy.inference.noise import (
    GaussianNoiseModel,
    StudentTNoiseModel,
    augment_configuration_with_noise,
)
from lyapy.likelihoods import gaussian_log_likelihood


def posterior_mean(sampler):
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    weights /= np.sum(weights)
    return np.average(sampler.results.samples, axis=0, weights=weights)


def make_hypotheses(entry, degrees_of_freedom):
    scaled_configuration, scaled = augment_configuration_with_noise(
        entry.configuration,
        include_jitter=False,
    )
    jitter_configuration, jitter = augment_configuration_with_noise(
        entry.configuration,
    )
    robust_configuration, robust = augment_configuration_with_noise(
        entry.configuration,
        distribution="student_t",
        degrees_of_freedom=degrees_of_freedom,
    )
    return (
        ModelHypothesis(
            "reported errors",
            entry.configuration,
            forward_model=entry.forward_model,
            likelihood_function=gaussian_log_likelihood,
        ),
        ModelHypothesis(
            "scaled errors",
            scaled_configuration,
            forward_model=entry.forward_model,
            likelihood_function=scaled,
        ),
        ModelHypothesis(
            "scaled errors + jitter",
            jitter_configuration,
            forward_model=entry.forward_model,
            likelihood_function=jitter,
        ),
        ModelHypothesis(
            "Student-t + jitter",
            robust_configuration,
            forward_model=entry.forward_model,
            likelihood_function=robust,
        ),
    )


def prediction_and_error(data, hypothesis, free_values):
    parameters = hypothesis.configuration.assemble_numpy(free_values)
    prediction = np.asarray(
        hypothesis.forward_model(
            data.wavelength,
            parameters,
            data.convolution_kernel,
        ).convolved.observed
    )
    likelihood = hypothesis.likelihood_function
    if isinstance(likelihood, (GaussianNoiseModel, StudentTNoiseModel)):
        effective_error = likelihood.effective_error_numpy(data.error, parameters)
    else:
        effective_error = data.error
    return prediction, effective_error


def fit_summary(data, prediction, effective_error):
    residuals = (data.flux - prediction) / effective_error
    return {
        "effective_error_scale_median": float(
            np.median(effective_error / data.error)
        ),
        "effective_error_minimum": float(np.min(effective_error)),
        "effective_error_median": float(np.median(effective_error)),
        "effective_error_maximum": float(np.max(effective_error)),
        "standardized_residual_root_mean_square": float(
            np.sqrt(np.mean(residuals**2))
        ),
        "maximum_absolute_standardized_residual": float(
            np.max(np.abs(residuals))
        ),
    }


def make_plot(data, selection):
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1, 1)},
    )
    profile_axis, residual_axis, uncertainty_axis = axes
    scale = 1.0e12
    profile_axis.errorbar(
        data.wavelength,
        data.flux * scale,
        yerr=data.error * scale,
        color="0.7",
        linewidth=0.8,
        label="GJ 411 STIS data",
    )
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    uncertainty_axis.axhline(1.0, color="0.4", linewidth=1.0)
    colors = plt.get_cmap("tab10").colors
    styles = ("-", "--", "-.", ":")
    for index, run in enumerate(selection.runs):
        hypothesis = run.hypothesis
        prediction, effective_error = prediction_and_error(
            data,
            hypothesis,
            posterior_mean(run.sampler),
        )
        color = colors[index]
        style = styles[index]
        profile_axis.plot(
            data.wavelength,
            prediction * scale,
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=hypothesis.name,
        )
        residual_axis.plot(
            data.wavelength,
            (data.flux - prediction) / effective_error,
            color=color,
            linestyle=style,
        )
        uncertainty_axis.plot(
            data.wavelength,
            effective_error / data.error,
            color=color,
            linestyle=style,
            label=hypothesis.name,
        )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False, ncol=2)
    profile_axis.grid(alpha=0.2)
    residual_axis.set_ylabel("residual / effective error")
    residual_axis.grid(alpha=0.2)
    uncertainty_axis.set_xlabel(r"wavelength ($\AA$)")
    uncertainty_axis.set_ylabel("effective / reported error")
    uncertainty_axis.grid(alpha=0.2)
    figure.suptitle(
        "GJ 411 uncertainty-model comparison: "
        f"best = {selection.best_model}"
    )
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stellar-model", default="voigt_legacy_reversal")
    parser.add_argument("--student-t-dof", type=float, default=4.0)
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
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("gj411_noise_model_comparison"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    problem = load_gj411_composable_problem(
        oversample=arguments.oversample,
        model_padding=arguments.model_padding,
    )
    catalog = build_gj411_model_catalog(
        problem.data.wavelength,
        oversample=arguments.oversample,
        model_padding=arguments.model_padding,
    )
    if arguments.stellar_model not in catalog:
        raise SystemExit(
            f"unknown --stellar-model {arguments.stellar_model!r}; "
            f"choices: {', '.join(catalog)}"
        )
    entry = catalog[arguments.stellar_model]
    hypotheses = make_hypotheses(entry, arguments.student_t_dof)
    selection = run_evidence_comparison(
        problem.data,
        hypotheses,
        random_seed=arguments.seed,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        print_progress=not arguments.no_progress,
        sample=arguments.dynesty_sample,
        bootstrap=0,
    )
    model_reports = {}
    for run in selection.runs:
        names = tuple(
            parameter.name
            for parameter in run.hypothesis.configuration.free_parameters
        )
        prediction, effective_error = prediction_and_error(
            problem.data,
            run.hypothesis,
            posterior_mean(run.sampler),
        )
        model_reports[run.hypothesis.name] = {
            "hypothesis": run.hypothesis.as_dict(),
            "posterior": summarize_dynesty(run.sampler, names),
            "fit": fit_summary(problem.data, prediction, effective_error),
        }
    report = {
        "problem": "observed GJ 411 uncertainty-model comparison",
        "stellar_model": entry.as_dict(),
        "seed": arguments.seed,
        "settings": {
            "live_points": arguments.nlive,
            "stopping_log_evidence": arguments.dlogz,
            "dynesty_sample": arguments.dynesty_sample,
            "student_t_degrees_of_freedom": arguments.student_t_dof,
        },
        "model_selection": selection.as_dict(),
        "models": model_reports,
    }
    figure = make_plot(problem.data, selection)
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["model_selection"]["ranking"], indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
