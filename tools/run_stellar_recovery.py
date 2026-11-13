"""Run five-parameter stellar-emission recovery with NUTS and dynesty."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

try:
    import corner
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
    import numpyro  # noqa: F401 -- fail early with an actionable message
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. From the repository root, run:\n\n"
        "    python -m pip install -e '.[inference]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import GJ411_STELLAR_CONFIGURATION, SpectrumData
from lyapy.inference.diagnostics import (
    summarize_dynesty,
    summarize_nuts,
    weighted_quantile,
)
from lyapy.inference.dynesty_backend import run_dynesty
from lyapy.inference.numpyro_backend import run_nuts
from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.observations import load_gj411_observation


PARAMETER_NAMES = tuple(
    parameter.name for parameter in GJ411_STELLAR_CONFIGURATION.free_parameters
)
PARAMETER_LABELS = (
    "stellar velocity",
    "log amplitude",
    "Lorentz FWHM",
    "Gaussian FWHM",
    "reversal depth",
)


def make_synthetic_problem(
    seed,
    configuration=GJ411_STELLAR_CONFIGURATION,
    noise_standard_deviation=5e-14,
):
    """Generate one reproducible noisy spectrum for a parameter configuration."""
    observation = load_gj411_observation()
    parameters = np.asarray(configuration.base_parameters, dtype=np.float64)
    truth_flux = np.asarray(
        evaluate_voigt_lya_model_jax(
            observation.wavelength,
            parameters,
            observation.convolution_kernel,
        ).convolved.observed
    )
    error = np.full_like(truth_flux, noise_standard_deviation)
    random = np.random.default_rng(seed)
    data = SpectrumData(
        observation.wavelength,
        truth_flux + random.normal(scale=error),
        error,
        observation.convolution_kernel,
    )
    truth = {
        parameter.name: float(parameters[parameter.index])
        for parameter in configuration.free_parameters
    }
    return data, configuration, truth


def predict(data, configuration, free_values):
    """Evaluate the shared JAX model at one free-parameter vector."""
    parameters = configuration.assemble_jax(jnp.asarray(free_values))
    return np.asarray(
        evaluate_voigt_lya_model_jax(
            data.wavelength,
            parameters,
            data.convolution_kernel,
        ).convolved.observed
    )


def normalized_dynesty_weights(sampler):
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    return weights / np.sum(weights)


def make_corner_plot(
    nuts_samples,
    nested,
    truth=None,
    parameter_names=PARAMETER_NAMES,
    parameter_labels=PARAMETER_LABELS,
    title="Five-parameter synthetic GJ 411 recovery",
):
    """Overlay NUTS and dynesty joint posteriors."""
    truths = (
        None
        if truth is None
        else np.asarray([truth[name] for name in parameter_names])
    )
    nuts_matrix = np.column_stack(
        [np.asarray(nuts_samples[name]) for name in parameter_names]
    )
    weights = normalized_dynesty_weights(nested)
    ranges = []
    for index, name in enumerate(parameter_names):
        nuts_limits = np.quantile(nuts_matrix[:, index], (0.005, 0.995))
        dynesty_limits = weighted_quantile(
            nested.results.samples[:, index],
            (0.005, 0.995),
            weights,
        )
        values = (*nuts_limits, *dynesty_limits)
        if truth is not None:
            values += (truth[name],)
        lower = min(values)
        upper = max(values)
        padding = 0.08 * (upper - lower)
        ranges.append((lower - padding, upper + padding))
    figure = corner.corner(
        nuts_matrix,
        labels=parameter_labels,
        truths=truths,
        color="#d6278b",
        truth_color="black",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        show_titles=True,
        range=ranges,
    )
    corner.corner(
        nested.results.samples,
        weights=weights,
        labels=parameter_labels,
        color="#1f77b4",
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.68, 0.95),
        range=ranges,
        fig=figure,
    )
    legend_handles = [
        Line2D([], [], color="#d6278b", label="NumPyro/NUTS"),
        Line2D([], [], color="#1f77b4", label="dynesty"),
    ]
    if truth is not None:
        legend_handles.append(
            Line2D([], [], color="black", linestyle="--", label="truth")
        )
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(title, y=1.01)
    return figure


def make_spectrum_plot(
    data,
    configuration,
    nuts_samples,
    nested,
    parameter_names=PARAMETER_NAMES,
    title="Five-parameter posterior representative curves",
    data_label="synthetic data",
):
    """Compare posterior representative curves with the fitted spectrum."""
    nuts_median = np.asarray(
        [np.median(nuts_samples[name]) for name in parameter_names]
    )
    dynesty_mean = np.average(
        nested.results.samples,
        axis=0,
        weights=normalized_dynesty_weights(nested),
    )
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    nuts_model = predict(data, configuration, nuts_median)
    dynesty_model = predict(data, configuration, dynesty_mean)
    profile_axis.errorbar(
        data.wavelength,
        data.flux * 1e12,
        yerr=data.error * 1e12,
        color="0.65",
        linewidth=0.8,
        label=data_label,
    )
    profile_axis.plot(
        data.wavelength,
        nuts_model * 1e12,
        color="#d6278b",
        linewidth=2.0,
        label="NUTS median",
    )
    profile_axis.plot(
        data.wavelength,
        dynesty_model * 1e12,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="dynesty mean",
    )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False)
    profile_axis.grid(alpha=0.2)
    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(
        data.wavelength,
        (data.flux - nuts_model) / data.error,
        color="#d6278b",
    )
    residual_axis.plot(
        data.wavelength,
        (data.flux - dynesty_model) / data.error,
        color="#1f77b4",
        linestyle="--",
    )
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / error")
    residual_axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def make_checks(nuts_summary, dynesty_summary, truth_available=True):
    r_hats = [
        parameter["r_hat"]
        for parameter in nuts_summary["parameters"].values()
        if parameter["r_hat"] is not None
    ]
    intervals_overlap = []
    for name, nuts_parameter in nuts_summary["parameters"].items():
        dynesty_parameter = dynesty_summary["parameters"][name]
        nuts_lower, nuts_upper = nuts_parameter["credible_interval"]
        dynesty_lower, dynesty_upper = dynesty_parameter["credible_interval"]
        intervals_overlap.append(
            max(nuts_lower, dynesty_lower) <= min(nuts_upper, dynesty_upper)
        )
    checks = {
        "nuts_no_divergences": nuts_summary["divergences"] == 0,
        "nuts_r_hat_at_most_1_01": bool(r_hats and max(r_hats) <= 1.01),
        "nuts_dynesty_intervals_overlap": all(intervals_overlap),
    }
    if truth_available:
        checks.update(
            nuts_truth_in_all_intervals=all(
                parameter["truth_in_interval"]
                for parameter in nuts_summary["parameters"].values()
            ),
            dynesty_truth_in_all_intervals=all(
                parameter["truth_in_interval"]
                for parameter in dynesty_summary["parameters"].values()
            ),
        )
    return checks


def make_fit_summary(data, configuration, nuts_samples, nested, parameter_names):
    """Summarize representative posterior fits in residual units."""
    nuts_median = np.asarray(
        [np.median(nuts_samples[name]) for name in parameter_names]
    )
    dynesty_mean = np.average(
        nested.results.samples,
        axis=0,
        weights=normalized_dynesty_weights(nested),
    )
    degrees_of_freedom = data.wavelength.size - len(parameter_names)
    summary = {}
    representatives = (
        ("nuts_median", nuts_median),
        ("dynesty_mean", dynesty_mean),
    )
    for name, values in representatives:
        residuals = (data.flux - predict(data, configuration, values)) / data.error
        chi_squared = np.sum(residuals**2)
        summary[name] = {
            "chi_squared": float(chi_squared),
            "degrees_of_freedom": int(degrees_of_freedom),
            "reduced_chi_squared": float(chi_squared / degrees_of_freedom),
            "residual_root_mean_square": float(np.sqrt(np.mean(residuals**2))),
            "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        }
    return summary


def parse_arguments(default_output_prefix="stellar_recovery", description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
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
        default=Path(default_output_prefix),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main(
    configuration=GJ411_STELLAR_CONFIGURATION,
    parameter_labels=PARAMETER_LABELS,
    problem="synthetic GJ 411 five-parameter stellar recovery",
    plot_title="Five-parameter synthetic GJ 411 recovery",
    default_output_prefix="stellar_recovery",
    description=__doc__,
    problem_factory=make_synthetic_problem,
    data_label="synthetic data",
):
    arguments = parse_arguments(default_output_prefix, description)
    parameter_names = tuple(
        parameter.name for parameter in configuration.free_parameters
    )
    prefix = arguments.output_prefix
    corner_path = prefix.with_name(prefix.name + "_corner.png")
    spectrum_path = prefix.with_name(prefix.name + "_spectrum.png")
    report_path = prefix.with_name(prefix.name + ".json")
    data, configuration, truth = problem_factory(
        arguments.seed,
        configuration,
    )

    start = perf_counter()
    nuts = run_nuts(
        data,
        configuration,
        jax.random.key(arguments.seed),
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        num_chains=arguments.chains,
        progress_bar=not arguments.no_progress,
        target_accept_probability=arguments.target_accept,
        chain_method="sequential",
        dense_mass=True,
    )
    nuts_seconds = perf_counter() - start
    start = perf_counter()
    nested = run_dynesty(
        data,
        configuration,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        random_state=np.random.default_rng(arguments.seed),
        print_progress=not arguments.no_progress,
        sample=arguments.dynesty_sample,
        bootstrap=0,
    )
    dynesty_seconds = perf_counter() - start

    nuts_summary = summarize_nuts(nuts, parameter_names, truth=truth)
    dynesty_summary = summarize_dynesty(nested, parameter_names, truth=truth)
    report = {
        "problem": problem,
        "seed": arguments.seed,
        "truth": truth,
        "priors": {
            parameter.name: [parameter.lower, parameter.upper]
            for parameter in configuration.free_parameters
        },
        "settings": {
            "nuts_warmup": arguments.warmup,
            "nuts_samples": arguments.samples,
            "nuts_chains": arguments.chains,
            "nuts_target_accept_probability": arguments.target_accept,
            "nuts_dense_mass": True,
            "dynesty_live_points": arguments.nlive,
            "dynesty_stopping_log_evidence": arguments.dlogz,
            "dynesty_sample": arguments.dynesty_sample,
        },
        "runtime_seconds": {"nuts": nuts_seconds, "dynesty": dynesty_seconds},
        "nuts": nuts_summary,
        "dynesty": dynesty_summary,
        "fit": make_fit_summary(
            data,
            configuration,
            nuts.get_samples(),
            nested,
            parameter_names,
        ),
        "checks": make_checks(
            nuts_summary,
            dynesty_summary,
            truth_available=truth is not None,
        ),
    }

    corner_figure = make_corner_plot(
        nuts.get_samples(),
        nested,
        truth,
        parameter_names=parameter_names,
        parameter_labels=parameter_labels,
        title=plot_title,
    )
    spectrum_figure = make_spectrum_plot(
        data,
        configuration,
        nuts.get_samples(),
        nested,
        parameter_names=parameter_names,
        title=plot_title + " curves",
        data_label=data_label,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corner_figure.savefig(corner_path, dpi=180, bbox_inches="tight")
    spectrum_figure.savefig(spectrum_path, dpi=180, bbox_inches="tight")
    plt.close(corner_figure)
    plt.close(spectrum_figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["checks"], indent=2))
    print(f"NUTS runtime:    {nuts_seconds:.2f} s")
    print(f"dynesty runtime: {dynesty_seconds:.2f} s")
    print(f"Wrote {corner_path.resolve()}")
    print(f"Wrote {spectrum_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
