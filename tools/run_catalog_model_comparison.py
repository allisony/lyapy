"""Compare any selected subset of the curated GJ 411 model catalog."""

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

from lyapy.inference import InferenceProblem, compare_inference_problems
from lyapy.inference.composable import load_gj411_composable_problem
from lyapy.inference.model_catalog import build_gj411_model_catalog


def fit_summary(data, result):
    model_flux = result.prediction_bands(probabilities=(0.5,))[0]
    residuals = (data.flux - model_flux) / data.error
    degrees_of_freedom = data.wavelength.size - len(result.parameter_names)
    chi_squared = np.sum(residuals**2)
    return {
        "chi_squared": float(chi_squared),
        "degrees_of_freedom": int(degrees_of_freedom),
        "reduced_chi_squared": float(chi_squared / degrees_of_freedom),
        "residual_root_mean_square": float(np.sqrt(np.mean(residuals**2))),
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
    }


def make_plot(data, entries, selection):
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
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
    colors = plt.get_cmap("tab10").colors
    line_styles = ("-", "--", "-.", ":")
    for index, entry in enumerate(entries):
        result = selection.get_result(entry.name)
        model_flux = result.prediction_bands(probabilities=(0.5,))[0]
        style = line_styles[index % len(line_styles)]
        color = colors[index % len(colors)]
        profile_axis.plot(
            data.wavelength,
            model_flux * scale,
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=entry.name.replace("_", " "),
        )
        residual_axis.plot(
            data.wavelength,
            (data.flux - model_flux) / data.error,
            color=color,
            linestyle=style,
            label=entry.name.replace("_", " "),
        )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False, ncol=2)
    profile_axis.grid(alpha=0.2)
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / error")
    residual_axis.grid(alpha=0.2)
    residual_axis.legend(frameon=False, ncol=2, fontsize="small")
    figure.suptitle(
        "GJ 411 stellar-model catalog comparison: "
        f"best = {selection.best_model.replace('_', ' ')}"
    )
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("single_gaussian", "single_voigt"),
        help="Two or more stable catalog model names",
    )
    parser.add_argument("--list-models", action="store_true")
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
        default=Path("gj411_catalog_comparison"),
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
    if arguments.list_models:
        for entry in catalog.values():
            print(f"{entry.name}: {entry.description}")
        return
    if len(arguments.models) < 2:
        raise SystemExit("--models requires at least two model names")
    if len(arguments.models) != len(set(arguments.models)):
        raise SystemExit("--models must not contain duplicates")
    missing = [name for name in arguments.models if name not in catalog]
    if missing:
        choices = ", ".join(catalog)
        raise SystemExit(
            f"unknown catalog model(s): {', '.join(missing)}; choices: {choices}"
        )
    entries = tuple(catalog[name] for name in arguments.models)
    problems = tuple(
        InferenceProblem(
            problem.data,
            entry.configuration,
            entry.forward_model,
            name=entry.name,
        )
        for entry in entries
    )
    selection = compare_inference_problems(
        problems,
        random_seed=arguments.seed,
        progress=not arguments.no_progress,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        sample=arguments.dynesty_sample,
        bootstrap=0,
    )
    model_reports = {}
    for entry in entries:
        result = selection.get_result(entry.name)
        model_reports[entry.name] = {
            "catalog": entry.as_dict(),
            "posterior": result.diagnostics,
            "fit": fit_summary(problem.data, result),
        }
    report = {
        "problem": "observed GJ 411 curated stellar-model comparison",
        "seed": arguments.seed,
        "selected_models": list(arguments.models),
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
        "models": model_reports,
    }
    figure = make_plot(problem.data, entries, selection)
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    comparison_directory = prefix.with_name(f"{prefix.name}_results")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest_path, _ = selection.save(comparison_directory)

    print(json.dumps(report["model_selection"]["ranking"], indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")
    print(f"Wrote portable results to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
