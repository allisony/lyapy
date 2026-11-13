"""Differentiable high-resolution convolution and detector-bin integration."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def detector_bin_edges(wavelength_centers):
    """Infer contiguous detector-bin edges from strictly increasing centers."""
    centers = np.asarray(wavelength_centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("wavelength_centers must be a vector with at least two values")
    if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0.0):
        raise ValueError("wavelength_centers must be finite and strictly increasing")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        (
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        )
    )


def flux_conserving_binning_matrix(high_resolution_edges, detector_edges):
    """Map piecewise-constant high-resolution flux density into detector bins."""
    high_edges = np.asarray(high_resolution_edges, dtype=np.float64)
    detector_edges = np.asarray(detector_edges, dtype=np.float64)
    if (
        high_edges.ndim != 1
        or detector_edges.ndim != 1
        or high_edges.size < 2
        or detector_edges.size < 2
    ):
        raise ValueError("bin edges must be one-dimensional")
    if np.any(np.diff(high_edges) <= 0.0) or np.any(np.diff(detector_edges) <= 0.0):
        raise ValueError("bin edges must be strictly increasing")
    left = np.maximum(detector_edges[:-1, None], high_edges[None, :-1])
    right = np.minimum(detector_edges[1:, None], high_edges[None, 1:])
    overlap = np.maximum(0.0, right - left)
    widths = np.diff(detector_edges)
    matrix = overlap / widths[:, None]
    if not np.allclose(np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("high-resolution grid must fully cover detector bins")
    return matrix


@dataclass(frozen=True)
class HighResolutionObservationOperator:
    """Static linear operator from a high-resolution model to detector bins."""

    wavelength: np.ndarray
    cell_edges: np.ndarray
    convolution_kernel: np.ndarray
    detector_edges: np.ndarray
    binning_matrix: np.ndarray
    oversample: int

    def apply(self, high_resolution_flux):
        """Convolve first, then integrate flux density into detector bins."""
        flux = jnp.asarray(high_resolution_flux)
        convolved = jnp.convolve(
            flux,
            jnp.asarray(self.convolution_kernel),
            mode="same",
        )
        detector_flux = jnp.asarray(self.binning_matrix) @ convolved
        return convolved, detector_flux


def build_high_resolution_operator(
    wavelength_centers,
    lsf_offsets,
    lsf_profile,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Build a flux-conserving operator for a shift-invariant tabulated LSF.

    ``lsf_offsets`` must be wavelength offsets in the same units as the data.
    The LSF is resampled on the internal uniform grid and normalized discretely.
    """
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    if model_padding < 0.0:
        raise ValueError("model_padding must be non-negative")
    detector_edges_array = detector_bin_edges(wavelength_centers)
    offsets = np.asarray(lsf_offsets, dtype=np.float64)
    profile = np.asarray(lsf_profile, dtype=np.float64)
    if offsets.ndim != 1 or profile.shape != offsets.shape or offsets.size < 2:
        raise ValueError("LSF offsets and profile must be matching vectors")
    if np.any(np.diff(offsets) <= 0.0):
        raise ValueError("LSF offsets must be strictly increasing")
    if np.any(profile < 0.0) or not np.any(profile > 0.0):
        raise ValueError("LSF profile must be non-negative and nonzero")

    detector_spacing = np.median(np.diff(np.asarray(wavelength_centers)))
    target_spacing = detector_spacing / oversample
    lsf_support = max(abs(offsets[0]), abs(offsets[-1]))
    padding = lsf_support + model_padding
    lower = detector_edges_array[0] - padding
    upper = detector_edges_array[-1] + padding
    number_of_cells = int(np.ceil((upper - lower) / target_spacing))
    high_edges = np.linspace(lower, upper, number_of_cells + 1)
    wavelength = 0.5 * (high_edges[:-1] + high_edges[1:])
    spacing = high_edges[1] - high_edges[0]

    kernel_radius = int(np.ceil(lsf_support / spacing))
    kernel_offsets = spacing * np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.interp(kernel_offsets, offsets, profile, left=0.0, right=0.0)
    kernel = kernel / np.sum(kernel)
    matrix = flux_conserving_binning_matrix(high_edges, detector_edges_array)
    return HighResolutionObservationOperator(
        wavelength=wavelength,
        cell_edges=high_edges,
        convolution_kernel=kernel,
        detector_edges=detector_edges_array,
        binning_matrix=matrix,
        oversample=oversample,
    )
