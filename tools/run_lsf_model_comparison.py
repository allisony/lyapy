"""Compare the calibrated STIS LSF with a fitted width-scale hypothesis."""

import argparse
import json
from pathlib import Path
import sys

try:
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. "
        "From the repository root, run:\n\n"
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
from lyapy.inference.lsf import augment_configuration_with_lsf_width
from lyapy.inference.model_catalog import build_gj411_model_catalog


def posterior_mean(sampler):
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    weights /= np.sum(weights)
    return np.average(sampler.results.samples, axis=0, weights=weights)


def make_hypotheses(entry, width_scale_bounds):
    fitted_configuration, fitted_forward_model = (
        augment_configuration_with_lsf_width(
            entry.configuration,
            entry.forward_model,
            width_scale_bounds=width_scale_bounds,
        )
    )
    return (
        ModelHypothesis(
            "calibrated LSF",
            entry.configuration,
            forward_model=entry.forward_model,
        ),
        ModelHypothesis(
            "fitted LSF width",
            fitted_configuration,
            forward_model=fitted_forward_model,
        ),
    )


def prediction(data, hypothesis, free_values):
    parameters = hypothesis.configuration.assemble_numpy(free_values)
    result = hypothesis.forward_model(
        data.wavelength,
        parameters,
        data.convolution_kernel,
    )
    return parameters, np.asarray(result.convolved.observed)


def lsf_curve(hypothesis, parameters, detector_spacing):
    operator = hypothesis.forward_model.observation_operator
    kernel = np.asarray(operator.effective_lsf_kernel(parameters))
    offsets_in_pixels = np.asarray(operator.kernel_offsets) / detector_spacing
    return offsets_in_pixels, kernel / np.max(kernel)


def make_plot(data, selection):
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.0),
        gridspec_kw={"height_ratios": (3, 1, 1.5)},
    )
    profile_axis, residual_axis, lsf_axis = axes
    scale = 1.0e12
    profile_axis.errorbar(
        data.wavelength,
        data.flux * scale,
        yerr=data.error * scale,
        color="0.72",
        linewidth=0.8,
        label="GJ 411 STIS data",
    )
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    colors = plt.get_cmap("tab10").colors
    detector_spacing = float(np.median(np.diff(data.wavelength)))
    for index, run in enumerate(selection.runs):
        hypothesis = run.hypothesis
        parameters, model_flux = prediction(
            data,
            hypothesis,
            posterior_mean(run.sampler),
        )
        color = colors[index]
        profile_axis.plot(
            data.wavelength,
            model_flux * scale,
            color=color,
            linewidth=1.8,
            label=hypothesis.name,
        )
        residual_axis.plot(
            data.wavelength,
            (data.flux - model_flux) / data.error,
            color=color,
            linewidth=1.3,
            label=hypothesis.name,
        )
        offsets, kernel = lsf_curve(hypothesis, parameters, detector_spacing)
        lsf_axis.plot(
            offsets,
            kernel,
            color=color,
            linewidth=1.8,
            label=hypothesis.name,
        )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False)
    profile_axis.grid(alpha=0.2)
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / error")
    residual_axis.grid(alpha=0.2)
    lsf_axis.set_xlabel("LSF offset (detector pixels)")
    lsf_axis.set_ylabel("relative response")
    lsf_axis.set_xlim(-5.0, 5.0)
    lsf_axis.set_yscale("log")
    lsf_axis.set_ylim(1e-3, 1.2)
    lsf_axis.grid(alpha=0.2)
    lsf_axis.legend(frameon=False)
    figure.suptitle(
        "GJ 411 STIS LSF comparison: "
        f"best = {selection.best_model}"
    )
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stellar-model", default="voigt_legacy_reversal")
    parser.add_argument("--minimum-width-scale", type=float, default=0.8)
    parser.add_argument("--maximum-width-scale", type=float, default=1.25)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--model-padding", type=float, default=1.0)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--dlogz", type=float, default=0.05)
    parser.add_argument(
        "--dynesty-sample",
        choices=("rwalk", "rslice"),
        default="rwalk",
    )
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("gj411_lsf_model_comparison"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    width_scale_bounds = (
        arguments.minimum_width_scale,
        arguments.maximum_width_scale,
    )
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
    hypotheses = make_hypotheses(entry, width_scale_bounds)
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
        free_values = posterior_mean(run.sampler)
        parameters, model_flux = prediction(
            problem.data,
            run.hypothesis,
            free_values,
        )
        names = tuple(
            parameter.name
            for parameter in run.hypothesis.configuration.free_parameters
        )
        report = {
            "hypothesis": run.hypothesis.as_dict(),
            "posterior": summarize_dynesty(run.sampler, names),
            "standardized_residual_root_mean_square": float(
                np.sqrt(
                    np.mean(
                        ((problem.data.flux - model_flux) / problem.data.error) ** 2
                    )
                )
            ),
        }
        operator = run.hypothesis.forward_model.observation_operator
        if operator.ln_lsf_width_scale_index is None:
            report["posterior_mean_lsf_width_scale"] = 1.0
        else:
            report["posterior_mean_lsf_width_scale"] = float(
                np.exp(parameters[operator.ln_lsf_width_scale_index])
            )
        model_reports[run.hypothesis.name] = report
    report = {
        "problem": "observed GJ 411 STIS LSF-width comparison",
        "stellar_model": entry.as_dict(),
        "seed": arguments.seed,
        "settings": {
            "live_points": arguments.nlive,
            "stopping_log_evidence": arguments.dlogz,
            "dynesty_sample": arguments.dynesty_sample,
            "lsf_width_scale_bounds": list(width_scale_bounds),
            "lsf_parameterization": "LSF(offset / scale), discretely renormalized",
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
