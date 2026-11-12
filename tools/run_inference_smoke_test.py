"""Run a small NumPyro/dynesty parameter-recovery comparison."""

import argparse
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.dynesty_backend import run_dynesty
from lyapy.inference.numpyro_backend import run_nuts
from lyapy.jax_models import evaluate_voigt_lya_model_jax
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


def _make_synthetic_problem(seed):
    reference = evaluate_gj411_legacy_case()
    wave = np.asarray(reference["wave"], dtype=np.float64)
    kernel = np.asarray(reference["lsf_kernel"], dtype=np.float64)
    parameters = np.asarray(PARAMETERS, dtype=np.float64)
    truth_flux = np.asarray(
        evaluate_voigt_lya_model_jax(wave, parameters, kernel).convolved.observed
    )
    error = np.full_like(truth_flux, 5e-14)
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
    truth = np.array([parameters[1], parameters[11]])
    return data, configuration, truth


def _predict(data, configuration, free_values):
    full_parameters = configuration.assemble_jax(jnp.asarray(free_values))
    return np.asarray(
        evaluate_voigt_lya_model_jax(
            data.wavelength,
            full_parameters,
            data.convolution_kernel,
        ).convolved.observed
    )


def _make_plot(
    data,
    configuration,
    truth,
    nuts_samples,
    dynesty_samples,
    dynesty_weights,
):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ("log amplitude", "reversal depth")
    names = ("log_amplitude", "reversal_depth")
    for index, (axis, label, name) in enumerate(zip(axes[:2], labels, names)):
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
        axis.axvline(truth[index], color="black", linestyle="--", label="truth")
        axis.set_xlabel(label)
        axis.set_ylabel("posterior density")
        axis.grid(alpha=0.2)

    nuts_median = np.array([np.median(nuts_samples[name]) for name in names])
    dynesty_mean = np.average(dynesty_samples, axis=0, weights=dynesty_weights)
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
        _predict(data, configuration, nuts_median) * 1e12,
        color="#d6278b",
        label="NUTS median",
    )
    axes[2].plot(
        data.wavelength,
        _predict(data, configuration, dynesty_mean) * 1e12,
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
    figure.suptitle("Synthetic GJ 411 inference smoke test")
    figure.tight_layout()
    return figure, nuts_median, dynesty_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--dlogz", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inference_smoke_test.png"),
    )
    arguments = parser.parse_args()

    data, configuration, truth = _make_synthetic_problem(arguments.seed)
    nuts = run_nuts(
        data,
        configuration,
        jax.random.key(arguments.seed),
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
    )
    nested = run_dynesty(
        data,
        configuration,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        random_state=np.random.default_rng(arguments.seed),
    )

    nuts_samples = nuts.get_samples()
    dynesty_samples = nested.results.samples
    dynesty_weights = np.exp(
        nested.results.logwt - nested.results.logz[-1]
    )
    figure, nuts_median, dynesty_mean = _make_plot(
        data,
        configuration,
        truth,
        nuts_samples,
        dynesty_samples,
        dynesty_weights,
    )
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Truth:         {truth}")
    print(f"NUTS median:   {nuts_median}")
    print(f"dynesty mean:  {dynesty_mean}")
    print(f"log evidence:  {nested.results.logz[-1]:.3f}")
    print(f"Wrote {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
