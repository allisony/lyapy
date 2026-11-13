"""Posterior Ly-alpha reconstruction products."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lyapy.inference.diagnostics import weighted_quantile
from lyapy.jax_models import evaluate_voigt_lya_model_jax


DEFAULT_PERCENTILE_PROBABILITIES = (0.025, 0.159, 0.5, 0.841, 0.975)


class ReconstructionSamples(NamedTuple):
    """Profile and integrated-flux values for posterior parameter draws."""

    intrinsic: np.ndarray
    convolved_observed: np.ndarray
    integrated_intrinsic_flux: np.ndarray


def evaluate_reconstruction_samples(data, configuration, free_values):
    """Evaluate reconstructed profiles for an array of free-parameter draws."""
    values = np.asarray(free_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != configuration.ndim:
        raise ValueError(
            "free_values must have shape (draws, configuration.ndim)"
        )
    wave = jnp.asarray(data.wavelength)
    kernel = jnp.asarray(data.convolution_kernel)

    def evaluate_one(draw):
        parameters = configuration.assemble_jax(draw)
        result = evaluate_voigt_lya_model_jax(wave, parameters, kernel)
        intrinsic = result.unconvolved.intrinsic
        integrated_flux = jnp.trapezoid(intrinsic, wave)
        return intrinsic, result.convolved.observed, integrated_flux

    intrinsic, convolved_observed, integrated_flux = jax.jit(
        jax.vmap(evaluate_one)
    )(jnp.asarray(values))
    return ReconstructionSamples(
        intrinsic=np.asarray(intrinsic),
        convolved_observed=np.asarray(convolved_observed),
        integrated_intrinsic_flux=np.asarray(integrated_flux),
    )


def posterior_percentile_bands(
    profile_samples,
    probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    weights=None,
):
    """Return wavelength-by-wavelength posterior percentile bands."""
    samples = np.asarray(profile_samples, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("profile_samples must have shape (draws, wavelength)")
    if weights is None:
        return np.quantile(samples, probabilities, axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (samples.shape[0],):
        raise ValueError("weights must contain one value per posterior draw")
    return np.column_stack(
        [
            weighted_quantile(samples[:, index], probabilities, weights)
            for index in range(samples.shape[1])
        ]
    )


def summarize_integrated_flux(
    integrated_flux_samples,
    probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    weights=None,
):
    """Summarize the posterior integrated intrinsic line flux."""
    values = np.asarray(integrated_flux_samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("integrated_flux_samples must be a non-empty vector")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    quantiles = weighted_quantile(values, probabilities, weights)
    if weights is None:
        mean = np.mean(values)
        standard_deviation = np.std(values)
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        normalized_weights = normalized_weights / np.sum(normalized_weights)
        mean = np.sum(normalized_weights * values)
        standard_deviation = np.sqrt(
            np.sum(normalized_weights * (values - mean) ** 2)
        )
    return {
        "mean": float(mean),
        "standard_deviation": float(standard_deviation),
        "percentile_probabilities": probabilities.tolist(),
        "percentiles": quantiles.tolist(),
    }
