"""Explicit prior calculations shared by legacy fitting entry points."""

import numpy as np


def legacy_bounded_log_prior(
    values,
    bounds,
    gaussian_flags,
    gaussian_parameters,
    *,
    parameter_names=None,
    log_weight_parameters=(),
):
    """Evaluate the historical bounded/Gaussian prior convention.

    Gaussian terms intentionally omit their normalization constants to preserve
    the legacy log-posterior. Parameters named in ``log_weight_parameters``
    receive the historical ``+log(value)`` term when their Gaussian flag is
    false. All remaining in-bounds parameters have a uniform log density of
    zero, again matching the legacy convention.
    """
    values = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    gaussian_flags = np.asarray(gaussian_flags, dtype=bool)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if bounds.shape != (values.size, 2):
        raise ValueError("bounds must have shape (number of values, 2)")
    if gaussian_flags.shape != values.shape:
        raise ValueError("gaussian_flags must match values")
    if len(gaussian_parameters) != values.size:
        raise ValueError("gaussian_parameters must match values")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("every lower bound must be less than its upper bound")
    if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
        return -np.inf

    if parameter_names is None:
        parameter_names = (None,) * values.size
    if len(parameter_names) != values.size:
        raise ValueError("parameter_names must match values")

    log_prior = 0.0
    for index, (value, use_gaussian, name) in enumerate(
        zip(values, gaussian_flags, parameter_names)
    ):
        if use_gaussian:
            mean, standard_deviation = gaussian_parameters[index]
            if standard_deviation <= 0.0:
                raise ValueError("Gaussian standard deviations must be positive")
            log_prior += -0.5 * ((value - mean) / standard_deviation) ** 2
        elif name in log_weight_parameters:
            if value <= 0.0:
                return -np.inf
            log_prior += np.log(value)
    return float(log_prior)
