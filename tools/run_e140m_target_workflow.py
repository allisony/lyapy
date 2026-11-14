"""Run and predictively compare NUTS fits for reviewed E140M spectra."""

import argparse
import json
from pathlib import Path

try:
    import jax
    import numpyro  # noqa: F401 -- fail early with an actionable message
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing NUTS dependency {error.name!r}. From the repository root, run:\n\n"
        "    python -m pip install -e '.[nuts]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from lyapy.inference import (
    InferenceProblem,
    build_e140m_target_model_catalog,
    compare_nuts_results,
    load_e140m_target_data,
)
from lyapy.observations import GJ411_E140M_TARGET, GJ729_E140M_TARGET


TARGETS = {
    "gj411": GJ411_E140M_TARGET,
    "gj729": GJ729_E140M_TARGET,
}


def _initial_log_total_flux(data):
    """Return a finite initialization heuristic, not a fitted-data prior."""
    observed_positive_flux = np.trapezoid(
        np.clip(data.flux, 0.0, None),
        data.wavelength,
    )
    if not np.isfinite(observed_positive_flux) or observed_positive_flux <= 0.0:
        return -11.75
    # The intrinsic profile must exceed the absorbed detector-space integral.
    # This affects only sampler initialization; the declared flux prior remains
    # the same broad range for both targets and all catalog models.
    return float(np.log10(1.7 * observed_positive_flux))


def _slug(value):
    return "_".join(str(value).lower().split())


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(TARGETS),
        default=tuple(TARGETS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=("single_gaussian", "single_voigt"),
        help="One or more stable curated model names",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--model-padding", type=float, default=1.0)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("e140m-target-workflow"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short mechanics check; not a scientifically converged fit",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if len(arguments.models) >= 2 and not arguments.list_models:
        try:
            import arviz  # noqa: F401 -- fail before running expensive fits
        except ModuleNotFoundError as error:
            raise SystemExit(
                "Comparing two or more NUTS models requires ArviZ. From the "
                "repository root, run:\n\n"
                "    python -m pip install -e '.[comparison]'"
            ) from error
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = {}

    for target_index, target_key in enumerate(arguments.targets):
        target = TARGETS[target_key]
        data = load_e140m_target_data(target)
        initial_log_total_flux = _initial_log_total_flux(data)
        catalog = build_e140m_target_model_catalog(
            target,
            data.wavelength,
            initial_log_total_flux=initial_log_total_flux,
            oversample=4 if arguments.quick else arguments.oversample,
            model_padding=arguments.model_padding,
        )
        if arguments.list_models:
            if target_index == 0:
                for entry in catalog.values():
                    print(f"{entry.name}: {entry.description}")
            continue
        missing = [name for name in arguments.models if name not in catalog]
        if missing:
            choices = ", ".join(catalog)
            raise SystemExit(
                f"unknown model(s): {', '.join(missing)}; choices: {choices}"
            )

        target_directory = output_directory / target_key
        target_directory.mkdir(parents=True, exist_ok=True)
        model_reports = {}
        completed_results = []
        for model_index, model_name in enumerate(arguments.models):
            entry = catalog[model_name]
            problem = InferenceProblem(
                data,
                entry.configuration,
                entry.forward_model,
                name=f"{target.name}: {entry.name}",
            )

            # Prior-predictive curves reveal implausible prior volume before
            # the expensive posterior fit. They remain noiseless latent curves.
            prior = problem.prior_predictive(
                draws=32 if arguments.quick else 500,
                random_seed=arguments.seed + 100 * target_index + model_index,
            )
            model_stem = target_directory / _slug(model_name)
            prior_plot = prior.plot()
            prior_plot_path = model_stem.with_name(
                f"{model_stem.name}_prior_predictive.png"
            )
            prior_plot.savefig(prior_plot_path, dpi=180)
            plt.close(prior_plot)

            # NUTS is the default inference backend. A quick run verifies only
            # mechanics; scientific fits require convergence assessment.
            result = problem.run(
                "nuts",
                random_seed=arguments.seed + 100 * target_index + model_index,
                progress=not arguments.no_progress,
                num_warmup=16 if arguments.quick else arguments.num_warmup,
                num_samples=16 if arguments.quick else arguments.num_samples,
                num_chains=arguments.num_chains,
                target_accept_probability=arguments.target_accept,
            )
            json_path, arrays_path = result.save(model_stem)
            fit_plot = result.plot(
                random_seed=arguments.seed + 1000 + model_index
            )
            fit_plot_path = model_stem.with_name(f"{model_stem.name}_fit.png")
            fit_plot.savefig(fit_plot_path, dpi=180)
            plt.close(fit_plot)
            completed_results.append(result)

            model_reports[model_name] = {
                "catalog": entry.as_dict(),
                "result": result.as_dict(),
                "fit_diagnostics": result.fit_diagnostics(
                    random_seed=arguments.seed + 1000 + model_index
                ).as_dict(),
                "prior_boundary_diagnostics": [
                    diagnostic.as_dict()
                    for diagnostic in result.prior_boundary_diagnostics()
                ],
                "files": {
                    "metadata": str(json_path),
                    "arrays": str(arrays_path),
                    "fit_plot": str(fit_plot_path),
                    "prior_predictive_plot": str(prior_plot_path),
                },
            }

        predictive_report = None
        if len(completed_results) >= 2:
            comparison = compare_nuts_results(completed_results)
            comparison_path = comparison.save(
                target_directory / "psis_loo_comparison"
            )
            comparison_plot = comparison.plot()
            comparison_plot_path = (
                target_directory / "psis_loo_comparison.png"
            )
            comparison_plot.savefig(comparison_plot_path, dpi=180)
            plt.close(comparison_plot)
            stacking_plot = comparison.plot_average()
            stacking_plot_path = target_directory / "predictive_stacking.png"
            stacking_plot.savefig(stacking_plot_path, dpi=180)
            plt.close(stacking_plot)
            pareto_paths = {}
            for model_name, loo_result in zip(
                arguments.models,
                comparison.loo_results,
            ):
                pareto_plot = loo_result.plot_pareto_k()
                pareto_path = target_directory / (
                    f"{_slug(model_name)}_pareto_k.png"
                )
                pareto_plot.savefig(pareto_path, dpi=180)
                plt.close(pareto_plot)
                pareto_paths[model_name] = str(pareto_path)
            predictive_report = comparison.as_dict()
            predictive_report["files"] = {
                "summary": str(comparison_path),
                "comparison_plot": str(comparison_plot_path),
                "stacking_plot": str(stacking_plot_path),
                "pareto_k_plots": pareto_paths,
            }

        reports[target_key] = {
            "target": target.as_dict(),
            "detector_pixels": int(data.wavelength.size),
            "fitted_pixels": data.fitted_pixels,
            "initial_log_total_flux": initial_log_total_flux,
            "models": model_reports,
            "predictive_comparison": predictive_report,
        }

    if arguments.list_models:
        return
    report = {
        "workflow": "two-target E140M NUTS fits and predictive comparison",
        "quick_mechanics_only": arguments.quick,
        "model_ranking": (
            "pixel-level PSIS-LOO with predictive stacking when at least two "
            "models are requested"
        ),
        "settings": {
            "oversample": 4 if arguments.quick else arguments.oversample,
            "model_padding_angstrom": arguments.model_padding,
            "num_warmup": 16 if arguments.quick else arguments.num_warmup,
            "num_samples": 16 if arguments.quick else arguments.num_samples,
            "num_chains": arguments.num_chains,
            "target_accept_probability": arguments.target_accept,
            "random_seed": arguments.seed,
        },
        "targets": reports,
    }
    report_path = output_directory / "workflow.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
