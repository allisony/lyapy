"""Posterior Ly-alpha reconstruction products."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lyapy.inference.diagnostics import weighted_quantile
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_models import evaluate_voigt_lya_model_jax


DEFAULT_PERCENTILE_PROBABILITIES = (0.025, 0.159, 0.5, 0.841, 0.975)


class ReconstructionSamples(NamedTuple):
    """Profile and integrated-flux values for posterior parameter draws."""

    model_wavelength: np.ndarray
    intrinsic: np.ndarray
    convolved_observed: np.ndarray
    integrated_intrinsic_flux: np.ndarray


class ReconstructionEvaluator:
    """Reusable compiled evaluator for one data/configuration/model triplet."""

    def __init__(
        self,
        data,
        configuration,
        forward_model=evaluate_voigt_lya_model_jax,
        batch_size=None,
    ):
        if batch_size is not None and (
            not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer or None")
        self.configuration = configuration
        self.wave = jnp.asarray(data.wavelength)
        self.kernel = jnp.asarray(data.convolution_kernel)
        if isinstance(forward_model, ComposableBinnedModel):
            self.model_wavelength = jnp.asarray(
                forward_model.observation_operator.wavelength
            )
            has_direct_total_flux = True
        else:
            baseline_result = forward_model(
                self.wave,
                jnp.asarray(configuration.base_parameters),
                self.kernel,
            )
            self.model_wavelength = jnp.asarray(
                getattr(baseline_result, "model_wavelength", self.wave)
            )
            has_direct_total_flux = hasattr(
                baseline_result.unconvolved,
                "total_flux",
            )

        def evaluate_one(draw):
            parameters = configuration.assemble_jax(draw)
            result = forward_model(self.wave, parameters, self.kernel)
            intrinsic = result.unconvolved.intrinsic
            if has_direct_total_flux:
                integrated_flux = result.unconvolved.total_flux
            else:
                integrated_flux = jnp.trapezoid(
                    intrinsic,
                    self.model_wavelength,
                )
            return intrinsic, result.convolved.observed, integrated_flux

        evaluate_many = (
            jax.vmap(evaluate_one)
            if batch_size is None
            else lambda draws: jax.lax.map(
                evaluate_one,
                draws,
                batch_size=batch_size,
            )
        )
        self._compiled_evaluate = jax.jit(evaluate_many)

    def evaluate(self, free_values):
        """Evaluate a matrix of posterior draws using the compiled batch."""
        values = np.asarray(free_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.configuration.ndim:
            raise ValueError(
                "free_values must have shape (draws, configuration.ndim)"
            )
        intrinsic, convolved_observed, integrated_flux = (
            self._compiled_evaluate(jnp.asarray(values))
        )
        return ReconstructionSamples(
            model_wavelength=np.asarray(self.model_wavelength),
            intrinsic=np.asarray(intrinsic),
            convolved_observed=np.asarray(convolved_observed),
            integrated_intrinsic_flux=np.asarray(integrated_flux),
        )


def evaluate_reconstruction_samples(
    data,
    configuration,
    free_values,
    forward_model=evaluate_voigt_lya_model_jax,
    batch_size=None,
):
    """Evaluate profiles, creating a temporary compiled evaluator."""
    return ReconstructionEvaluator(
        data,
        configuration,
        forward_model=forward_model,
        batch_size=batch_size,
    ).evaluate(free_values)


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
