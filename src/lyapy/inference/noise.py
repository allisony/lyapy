"""Explicit uncertainty models for spectral likelihoods."""

from dataclasses import asdict, dataclass
import inspect
import math

import jax.numpy as jnp
from jax.scipy.special import gammaln
import numpy as np

from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.likelihoods import gaussian_log_likelihood


def _validate_index(index, name):
    if index is not None and index < 0:
        raise ValueError(f"{name} must be non-negative")


def _callable_accepts_parameters(function):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    parameters = tuple(signature.parameters.values())
    return any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    ) or len(parameters) >= 4


def evaluate_numpy_log_likelihood(
    likelihood_function,
    observed,
    predicted,
    error,
    parameters,
):
    """Evaluate a legacy callable or parameter-aware noise model with NumPy."""
    method = getattr(likelihood_function, "numpy_log_likelihood", None)
    if method is not None:
        return float(method(observed, predicted, error, parameters))
    if _callable_accepts_parameters(likelihood_function):
        return float(likelihood_function(observed, predicted, error, parameters))
    return float(likelihood_function(observed, predicted, error))


def evaluate_jax_log_likelihood(
    likelihood_function,
    observed,
    predicted,
    error,
    parameters,
):
    """Evaluate a legacy Gaussian or parameter-aware model with JAX."""
    method = getattr(likelihood_function, "jax_log_likelihood", None)
    if method is not None:
        return method(observed, predicted, error, parameters)
    if likelihood_function is gaussian_log_likelihood:
        effective_error = jnp.asarray(error)
        residual = (jnp.asarray(observed) - jnp.asarray(predicted)) / effective_error
        return -0.5 * jnp.sum(
            residual**2 + jnp.log(2.0 * jnp.pi * effective_error**2)
        )
    if _callable_accepts_parameters(likelihood_function):
        return likelihood_function(observed, predicted, error, parameters)
    return likelihood_function(observed, predicted, error)


@dataclass(frozen=True)
class GaussianNoiseModel:
    """Gaussian likelihood with optional scale and additive jitter parameters."""

    ln_error_scale_index: int | None = None
    log10_jitter_index: int | None = None
    fixed_error_scale: float = 1.0
    fixed_jitter: float = 0.0

    def __post_init__(self):
        _validate_index(self.ln_error_scale_index, "ln_error_scale_index")
        _validate_index(self.log10_jitter_index, "log10_jitter_index")
        if self.fixed_error_scale <= 0.0:
            raise ValueError("fixed_error_scale must be positive")
        if self.fixed_jitter < 0.0:
            raise ValueError("fixed_jitter must be non-negative")

    def _scale_and_jitter_numpy(self, parameters):
        scale = (
            self.fixed_error_scale
            if self.ln_error_scale_index is None
            else np.exp(np.asarray(parameters)[self.ln_error_scale_index])
        )
        jitter = (
            self.fixed_jitter
            if self.log10_jitter_index is None
            else 10.0 ** np.asarray(parameters)[self.log10_jitter_index]
        )
        return scale, jitter

    def _scale_and_jitter_jax(self, parameters):
        scale = (
            self.fixed_error_scale
            if self.ln_error_scale_index is None
            else jnp.exp(parameters[self.ln_error_scale_index])
        )
        jitter = (
            self.fixed_jitter
            if self.log10_jitter_index is None
            else 10.0 ** parameters[self.log10_jitter_index]
        )
        return scale, jitter

    def effective_error_numpy(self, error, parameters):
        scale, jitter = self._scale_and_jitter_numpy(parameters)
        return np.sqrt((scale * np.asarray(error)) ** 2 + jitter**2)

    def effective_error_jax(self, error, parameters):
        scale, jitter = self._scale_and_jitter_jax(parameters)
        return jnp.sqrt((scale * jnp.asarray(error)) ** 2 + jitter**2)

    def numpy_log_likelihood(self, observed, predicted, error, parameters):
        effective_error = self.effective_error_numpy(error, parameters)
        return gaussian_log_likelihood(observed, predicted, effective_error)

    def jax_log_likelihood(self, observed, predicted, error, parameters):
        effective_error = self.effective_error_jax(error, parameters)
        residual = (jnp.asarray(observed) - jnp.asarray(predicted)) / effective_error
        return -0.5 * jnp.sum(
            residual**2 + jnp.log(2.0 * jnp.pi * effective_error**2)
        )

    def __call__(self, observed, predicted, error, parameters):
        return self.numpy_log_likelihood(observed, predicted, error, parameters)

    def as_dict(self):
        return {"distribution": "Gaussian", **asdict(self)}


@dataclass(frozen=True)
class StudentTNoiseModel(GaussianNoiseModel):
    """Independent Student-t residuals with optional scale and jitter."""

    degrees_of_freedom: float = 4.0

    def __post_init__(self):
        super().__post_init__()
        if self.degrees_of_freedom <= 0.0:
            raise ValueError("degrees_of_freedom must be positive")

    def numpy_log_likelihood(self, observed, predicted, error, parameters):
        effective_error = self.effective_error_numpy(error, parameters)
        residual = (np.asarray(observed) - np.asarray(predicted)) / effective_error
        nu = self.degrees_of_freedom
        normalization = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu * math.pi)
            - np.log(effective_error)
        )
        return float(
            np.sum(normalization - 0.5 * (nu + 1.0) * np.log1p(residual**2 / nu))
        )

    def jax_log_likelihood(self, observed, predicted, error, parameters):
        effective_error = self.effective_error_jax(error, parameters)
        residual = (jnp.asarray(observed) - jnp.asarray(predicted)) / effective_error
        nu = self.degrees_of_freedom
        normalization = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * jnp.log(nu * jnp.pi)
            - jnp.log(effective_error)
        )
        return jnp.sum(
            normalization - 0.5 * (nu + 1.0) * jnp.log1p(residual**2 / nu)
        )

    def as_dict(self):
        return {"distribution": "StudentT", **asdict(self)}


def augment_configuration_with_noise(
    configuration,
    *,
    include_error_scale=True,
    include_jitter=True,
    ln_error_scale_bounds=(math.log(0.5), math.log(5.0)),
    log10_jitter_bounds=(-16.0, -12.0),
    base_ln_error_scale=0.0,
    base_log10_jitter=-14.5,
    distribution="gaussian",
    degrees_of_freedom=4.0,
):
    """Append explicit nuisance parameters and return configuration plus model."""
    if not include_error_scale and not include_jitter:
        raise ValueError("at least one noise parameter must be included")
    names = list(configuration.parameter_names)
    base = list(configuration.base_parameters)
    free = list(configuration.free_parameters)
    scale_index = None
    jitter_index = None
    if include_error_scale:
        scale_index = len(names)
        names.append("ln_error_scale")
        base.append(base_ln_error_scale)
        free.append(
            UniformParameter(
                "ln_error_scale",
                scale_index,
                *ln_error_scale_bounds,
            )
        )
    if include_jitter:
        jitter_index = len(names)
        names.append("log10_jitter")
        base.append(base_log10_jitter)
        free.append(
            UniformParameter(
                "log10_jitter",
                jitter_index,
                *log10_jitter_bounds,
            )
        )
    augmented = InferenceConfiguration(
        base_parameters=tuple(base),
        free_parameters=tuple(free),
        parameter_names=tuple(names),
    )
    keyword_arguments = {
        "ln_error_scale_index": scale_index,
        "log10_jitter_index": jitter_index,
    }
    if distribution == "gaussian":
        noise_model = GaussianNoiseModel(**keyword_arguments)
    elif distribution == "student_t":
        noise_model = StudentTNoiseModel(
            **keyword_arguments,
            degrees_of_freedom=degrees_of_freedom,
        )
    else:
        raise ValueError("distribution must be 'gaussian' or 'student_t'")
    return augmented, noise_model
