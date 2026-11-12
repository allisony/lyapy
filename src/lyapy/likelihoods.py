"""Small, sampler-independent likelihood functions."""

import numpy as np


def gaussian_log_likelihood(observed, predicted, error):
    """Return the normalized independent-Gaussian log likelihood."""
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if observed.shape != predicted.shape or observed.shape != error.shape:
        raise ValueError("observed, predicted, and error must have matching shapes")
    if not all(
        np.all(np.isfinite(array)) for array in (observed, predicted, error)
    ):
        raise ValueError("likelihood inputs must contain only finite values")
    if np.any(error <= 0.0):
        raise ValueError("error values must be positive")
    residual = (observed - predicted) / error
    return float(
        -0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * error**2))
    )
