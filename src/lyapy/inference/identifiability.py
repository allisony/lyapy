"""Local JAX sensitivity and identifiability analysis for spectral models."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

from lyapy.inference.common import TruncatedNormalParameter, UniformParameter
from lyapy.jax_models import evaluate_voigt_lya_model_jax


def prior_standard_deviation(parameter):
    """Return the standard deviation implied by one supported prior."""
    if isinstance(parameter, UniformParameter):
        return (parameter.upper - parameter.lower) / np.sqrt(12.0)
    if isinstance(parameter, TruncatedNormalParameter):
        lower = (parameter.lower - parameter.mean) / parameter.standard_deviation
        upper = (parameter.upper - parameter.mean) / parameter.standard_deviation
        variance = truncnorm.var(
            lower,
            upper,
            loc=parameter.mean,
            scale=parameter.standard_deviation,
        )
        return float(np.sqrt(variance))
    raise TypeError(f"unsupported prior type: {type(parameter).__name__}")


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Local error-weighted geometry in prior-standardized coordinates."""

    parameter_names: tuple[str, ...]
    evaluation_values: np.ndarray
    prior_scales: np.ndarray
    jacobian: np.ndarray
    scaled_whitened_jacobian: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    posterior_covariance_scaled: np.ndarray
    posterior_correlation: np.ndarray
    numerical_rank: int

    @property
    def condition_number(self):
        """Return the ratio of largest to smallest local singular value."""
        if self.singular_values[-1] == 0.0:
            return np.inf
        return float(self.singular_values[0] / self.singular_values[-1])

    @property
    def posterior_to_prior_scale(self):
        """Return local posterior sigma divided by prior sigma per parameter."""
        return np.sqrt(np.diag(self.posterior_covariance_scaled))

    def strongest_correlations(self, minimum_absolute_correlation=0.8):
        """Return parameter pairs exceeding an absolute correlation threshold."""
        pairs = []
        for first in range(len(self.parameter_names)):
            for second in range(first + 1, len(self.parameter_names)):
                correlation = float(self.posterior_correlation[first, second])
                if abs(correlation) >= minimum_absolute_correlation:
                    pairs.append(
                        {
                            "first": self.parameter_names[first],
                            "second": self.parameter_names[second],
                            "correlation": correlation,
                        }
                    )
        return sorted(pairs, key=lambda pair: abs(pair["correlation"]), reverse=True)

    def weakest_direction(self):
        """Return loadings of the least data-constrained local direction."""
        loadings = self.right_singular_vectors[-1]
        return sorted(
            (
                {"parameter": name, "loading": float(loading)}
                for name, loading in zip(self.parameter_names, loadings)
            ),
            key=lambda item: abs(item["loading"]),
            reverse=True,
        )

    def as_dict(self):
        """Return a compact JSON-serializable analysis report."""
        sensitivity = np.linalg.norm(self.scaled_whitened_jacobian, axis=0)
        return {
            "analysis": "local linear error-weighted identifiability",
            "parameter_names": list(self.parameter_names),
            "evaluation_values": self.evaluation_values.tolist(),
            "prior_standard_deviations": self.prior_scales.tolist(),
            "prior_scaled_sensitivity_norms": sensitivity.tolist(),
            "singular_values": self.singular_values.tolist(),
            "numerical_rank": self.numerical_rank,
            "condition_number": self.condition_number,
            "posterior_to_prior_scale": self.posterior_to_prior_scale.tolist(),
            "posterior_correlation": self.posterior_correlation.tolist(),
            "weakest_direction": self.weakest_direction(),
            "strong_correlations": self.strongest_correlations(),
            "caveat": (
                "This is a local linear analysis at one parameter vector; it does "
                "not diagnose nonlinear degeneracy or global multimodality."
            ),
        }


def analyze_local_identifiability(data, configuration, free_values=None):
    """Differentiate the convolved model and summarize local parameter geometry."""
    if free_values is None:
        free_values = np.asarray(
            [
                configuration.base_parameters[parameter.index]
                for parameter in configuration.free_parameters
            ],
            dtype=np.float64,
        )
    else:
        free_values = np.asarray(free_values, dtype=np.float64)
    if free_values.shape != (configuration.ndim,):
        raise ValueError(
            f"expected {configuration.ndim} free values, got {free_values.shape}"
        )

    wavelength = jnp.asarray(data.wavelength, dtype=jnp.float64)
    kernel = jnp.asarray(data.convolution_kernel, dtype=jnp.float64)

    def predict(values):
        parameters = configuration.assemble_jax(values)
        return evaluate_voigt_lya_model_jax(
            wavelength,
            parameters,
            kernel,
        ).convolved.observed

    jacobian = np.asarray(jax.jacrev(predict)(jnp.asarray(free_values)))
    prior_scales = np.asarray(
        [prior_standard_deviation(parameter) for parameter in configuration.free_parameters]
    )
    scaled_whitened = (
        jacobian / np.asarray(data.error, dtype=np.float64)[:, None]
    ) * prior_scales[None, :]
    _, singular_values, right_singular_vectors = np.linalg.svd(
        scaled_whitened,
        full_matrices=False,
    )
    fisher_scaled = scaled_whitened.T @ scaled_whitened
    posterior_covariance = np.linalg.inv(
        fisher_scaled + np.eye(configuration.ndim)
    )
    posterior_standard_deviation = np.sqrt(np.diag(posterior_covariance))
    posterior_correlation = posterior_covariance / np.outer(
        posterior_standard_deviation,
        posterior_standard_deviation,
    )
    return IdentifiabilityResult(
        parameter_names=tuple(
            parameter.name for parameter in configuration.free_parameters
        ),
        evaluation_values=free_values,
        prior_scales=prior_scales,
        jacobian=jacobian,
        scaled_whitened_jacobian=scaled_whitened,
        singular_values=singular_values,
        right_singular_vectors=right_singular_vectors,
        posterior_covariance_scaled=posterior_covariance,
        posterior_correlation=posterior_correlation,
        numerical_rank=int(np.linalg.matrix_rank(scaled_whitened)),
    )
