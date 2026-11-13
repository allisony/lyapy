"""Differentiable high-resolution convolution and detector-bin integration."""

from dataclasses import dataclass, replace

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
    kernel_offsets: np.ndarray
    lsf_offsets: np.ndarray
    lsf_profile: np.ndarray
    model_padding: float
    ln_lsf_width_scale_index: int | None = None
    lsf_width_scale_bounds: tuple[float, float] = (1.0, 1.0)

    def effective_lsf_kernel(self, parameters=None):
        """Return the normalized fixed or parameter-scaled LSF kernel."""
        if self.ln_lsf_width_scale_index is None:
            return jnp.asarray(self.convolution_kernel)
        if parameters is None:
            raise ValueError("parameters are required for a fitted LSF width")
        width_scale = jnp.exp(
            jnp.asarray(parameters)[self.ln_lsf_width_scale_index]
        )
        kernel = jnp.interp(
            jnp.asarray(self.kernel_offsets) / width_scale,
            jnp.asarray(self.lsf_offsets),
            jnp.asarray(self.lsf_profile),
            left=0.0,
            right=0.0,
        )
        return kernel / jnp.sum(kernel)

    def apply(self, high_resolution_flux, parameters=None):
        """Convolve first, then integrate flux density into detector bins."""
        flux = jnp.asarray(high_resolution_flux)
        convolved = jnp.convolve(
            flux,
            self.effective_lsf_kernel(parameters),
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
    ln_lsf_width_scale_index=None,
    lsf_width_scale_bounds=(1.0, 1.0),
):
    """Build a flux-conserving operator for a shift-invariant tabulated LSF.

    ``lsf_offsets`` must be wavelength offsets in the same units as the data.
    The LSF is resampled on the internal uniform grid and normalized discretely.
    """
    if not isinstance(oversample, int) or oversample < 1:
        raise ValueError("oversample must be a positive integer")
    if model_padding < 0.0:
        raise ValueError("model_padding must be non-negative")
    if ln_lsf_width_scale_index is not None and ln_lsf_width_scale_index < 0:
        raise ValueError("ln_lsf_width_scale_index must be non-negative")
    minimum_width_scale, maximum_width_scale = lsf_width_scale_bounds
    if (
        minimum_width_scale <= 0.0
        or maximum_width_scale < minimum_width_scale
    ):
        raise ValueError("LSF width-scale bounds must be positive and ordered")
    if (
        ln_lsf_width_scale_index is not None
        and minimum_width_scale == maximum_width_scale
    ):
        raise ValueError("a fitted LSF width requires nonzero-width bounds")
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
    padding = maximum_width_scale * lsf_support + model_padding
    lower = detector_edges_array[0] - padding
    upper = detector_edges_array[-1] + padding
    number_of_cells = int(np.ceil((upper - lower) / target_spacing))
    high_edges = np.linspace(lower, upper, number_of_cells + 1)
    wavelength = 0.5 * (high_edges[:-1] + high_edges[1:])
    spacing = high_edges[1] - high_edges[0]

    kernel_radius = int(np.ceil(maximum_width_scale * lsf_support / spacing))
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
        kernel_offsets=kernel_offsets,
        lsf_offsets=offsets,
        lsf_profile=profile,
        model_padding=model_padding,
        ln_lsf_width_scale_index=ln_lsf_width_scale_index,
        lsf_width_scale_bounds=(minimum_width_scale, maximum_width_scale),
    )


def parameterize_lsf_width(
    operator,
    ln_lsf_width_scale_index,
    *,
    width_scale_bounds=(0.8, 1.25),
):
    """Copy an operator with a differentiable, flux-conserving LSF width."""
    if ln_lsf_width_scale_index < 0:
        raise ValueError("ln_lsf_width_scale_index must be non-negative")
    minimum_width_scale, maximum_width_scale = width_scale_bounds
    if minimum_width_scale <= 0.0 or maximum_width_scale <= minimum_width_scale:
        raise ValueError("width_scale_bounds must be positive and increasing")
    spacing = float(np.median(np.diff(operator.cell_edges)))
    support = max(abs(operator.lsf_offsets[0]), abs(operator.lsf_offsets[-1]))
    extra_support = max(0.0, maximum_width_scale - 1.0) * support
    if operator.model_padding < extra_support:
        raise ValueError(
            "model_padding is too small for the maximum fitted LSF width"
        )
    radius = int(np.ceil(maximum_width_scale * support / spacing))
    kernel_offsets = spacing * np.arange(-radius, radius + 1)
    kernel = np.interp(
        kernel_offsets,
        operator.lsf_offsets,
        operator.lsf_profile,
        left=0.0,
        right=0.0,
    )
    kernel /= np.sum(kernel)
    return replace(
        operator,
        convolution_kernel=kernel,
        kernel_offsets=kernel_offsets,
        ln_lsf_width_scale_index=ln_lsf_width_scale_index,
        lsf_width_scale_bounds=(minimum_width_scale, maximum_width_scale),
    )
