"""Validate NumPyro and dynesty recovery on a synthetic GJ 411 spectrum."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

try:
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
    import numpyro  # noqa: F401 -- fail early with an actionable message
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing optional inference dependency {error.name!r}.\n\n"
        "From the repository root, install the validated inference stack with:\n\n"
        "    python -m pip install -e '.[inference]'\n\n"
        "Then rerun this command."
    ) from error

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import (
    GJ411_CONFIGURATION,
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.diagnostics import summarize_dynesty, summarize_nuts
from lyapy.inference.dynesty_backend import run_dynesty
from lyapy.inference.numpyro_backend import run_nuts
from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.observations import load_gj411_observation


PARAMETER_NAMES = ("log_amplitude", "reversal_depth")
PARAMETER_LABELS = ("log amplitude", "reversal depth")


def make_synthetic_problem(seed, noise_standard_deviation=5e-14):
    """Return noisy synthetic data, priors, and generating parameters."""
    observation = load_gj411_observation()
    wave = observation.wavelength
    kernel = observation.convolution_kernel
    parameters = np.asarray(GJ411_CONFIGURATION.base_parameters, dtype=np.float64)
    truth_flux = np.asarray(
        evaluate_voigt_lya_model_jax(wave, parameters, kernel).convolved.observed
    )
    error = np.full_like(truth_flux, noise_standard_deviation)
    random = np.random.default_rng(seed)
    noisy_flux = truth_flux + random.normal(scale=error)
    data = SpectrumData(wave, noisy_flux, error, kernel)
    configuration = InferenceConfiguration(
        base_parameters=tuple(parameters),
        free_parameters=(
            UniformParameter("log_amplitude", 1, -10.3, -9.8),
            UniformParameter("reversal_depth", 11, 1.0, 1.9),
        ),
    )
    truth = {
        parameter.name: float(parameters[parameter.index])
        for parameter in configuration.free_parameters
    }
    return data, configuration, truth


def predict(data, configuration, free_values):
    """Evaluate the shared JAX model for a vector of free parameters."""
    full_parameters = configuration.assemble_jax(jnp.asarray(free_values))
    return np.asarray(
        evaluate_voigt_lya_model_jax(
            data.wavelength,
            full_parameters,
            data.convolution_kernel,
        ).convolved.observed
    )


def make_plot(
    data,
    configuration,
    truth,
    nuts_samples,
    dynesty_samples,
    dynesty_weights,
    interval_probability,
):
    """Plot the posterior comparison and representative reconstructed curves."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for index, (axis, label, name) in enumerate(
        zip(axes[:2], PARAMETER_LABELS, PARAMETER_NAMES)
    ):
        axis.hist(
            nuts_samples[name],
            bins=35,
            density=True,
            histtype="step",
            linewidth=2,
            color="#d6278b",
            label="NumPyro/NUTS",
        )
        axis.hist(
            dynesty_samples[:, index],
            bins=35,
            weights=dynesty_weights,
            density=True,
            histtype="step",
            linewidth=2,
            color="#1f77b4",
            label="dynesty",
        )
        axis.axvline(truth[name], color="black", linestyle="--", label="truth")
        axis.set_xlabel(label)
        axis.set_ylabel("posterior density")
        axis.grid(alpha=0.2)

    nuts_median = np.array(
        [np.median(nuts_samples[name]) for name in PARAMETER_NAMES]
    )
    dynesty_mean = np.average(
        dynesty_samples,
        axis=0,
        weights=dynesty_weights,
    )
    axes[2].errorbar(
        data.wavelength,
        data.flux * 1e12,
        yerr=data.error * 1e12,
        color="0.65",
        linewidth=0.8,
        label="synthetic data",
    )
    axes[2].plot(
        data.wavelength,
        predict(data, configuration, nuts_median) * 1e12,
        color="#d6278b",
        label="NUTS median",
    )
    axes[2].plot(
        data.wavelength,
        predict(data, configuration, dynesty_mean) * 1e12,
        color="#1f77b4",
        linestyle="--",
        label="dynesty mean",
    )
    axes[2].set_xlabel(r"wavelength ($\AA$)")
    axes[2].set_ylabel(r"flux density ($10^{-12}$ units)")
    axes[2].xaxis.set_major_locator(MaxNLocator(6))
    axes[2].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    figure.suptitle(
        f"Synthetic GJ 411 recovery ({interval_probability:.0%} intervals)"
    )
    figure.tight_layout()
    return figure


def make_checks(nuts_summary, dynesty_summary):
    """Return simple validation checks without hiding the underlying metrics."""
    r_hats = [
        parameter["r_hat"]
        for parameter in nuts_summary["parameters"].values()
        if parameter["r_hat"] is not None
    ]
    return {
        "nuts_no_divergences": nuts_summary["divergences"] == 0,
        "nuts_r_hat_at_most_1_01": (
            None if not r_hats else bool(max(r_hats) <= 1.01)
        ),
        "nuts_truth_in_all_intervals": all(
            parameter["truth_in_interval"]
            for parameter in nuts_summary["parameters"].values()
        ),
        "dynesty_truth_in_all_intervals": all(
            parameter["truth_in_interval"]
            for parameter in dynesty_summary["parameters"].values()
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--nlive", type=int, default=300)
    parser.add_argument("--dlogz", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--interval", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_validation.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON destination (defaults to the plot name with a .json suffix)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress sampler progress displays",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    report_path = (
        arguments.report
        if arguments.report is not None
        else arguments.output.with_suffix(".json")
    )
    data, configuration, truth = make_synthetic_problem(arguments.seed)

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
    )
    dynesty_seconds = perf_counter() - start

    nuts_summary = summarize_nuts(
        nuts,
        PARAMETER_NAMES,
        truth=truth,
        interval_probability=arguments.interval,
    )
    dynesty_summary = summarize_dynesty(
        nested,
        PARAMETER_NAMES,
        truth=truth,
        interval_probability=arguments.interval,
    )
    report = {
        "problem": "synthetic GJ 411 two-parameter recovery",
        "seed": arguments.seed,
        "credible_interval_probability": arguments.interval,
        "truth": truth,
        "settings": {
            "nuts_warmup": arguments.warmup,
            "nuts_samples": arguments.samples,
            "nuts_chains": arguments.chains,
            "nuts_target_accept_probability": arguments.target_accept,
            "dynesty_live_points": arguments.nlive,
            "dynesty_stopping_log_evidence": arguments.dlogz,
        },
        "runtime_seconds": {
            "nuts": nuts_seconds,
            "dynesty": dynesty_seconds,
        },
        "nuts": nuts_summary,
        "dynesty": dynesty_summary,
        "checks": make_checks(nuts_summary, dynesty_summary),
    }

    nuts_samples = nuts.get_samples()
    dynesty_weights = np.exp(
        nested.results.logwt - nested.results.logz[-1]
    )
    dynesty_weights = dynesty_weights / np.sum(dynesty_weights)
    figure = make_plot(
        data,
        configuration,
        truth,
        nuts_samples,
        nested.results.samples,
        dynesty_weights,
        arguments.interval,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["checks"], indent=2))
    print(f"NUTS runtime:    {nuts_seconds:.2f} s")
    print(f"dynesty runtime: {dynesty_seconds:.2f} s")
    print(f"Wrote {arguments.output.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
