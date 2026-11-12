"""Instrument-resolution kernels and STIS line-spread functions."""

import numpy as np


def gaussian_kernel(grid, fwhm, nfwhm=4.0):
    """Return the normalized legacy Gaussian convolution kernel."""
    grid = np.asarray(grid)
    spacing = (grid[-1] - grid[0]) / (len(grid) - 1.0)
    kernel_points = round(nfwhm * fwhm / spacing)
    if kernel_points % 2 != 0:
        kernel_points += 1
    offsets = spacing * (np.arange(kernel_points) - kernel_points / 2.0)
    kernel = np.exp(-np.log(2.0) / (fwhm / 2.0) ** 2 * offsets**2)
    normalized = kernel / np.sum(kernel)
    return np.append(normalized, normalized[0])


def interpolate_stis_lsf(
    original_lsf_wavelength,
    original_lsf,
    grating_dispersion,
    data_wavelength,
):
    """Interpolate and normalize a tabulated STIS line-spread function."""
    data_spacing = data_wavelength[1] - data_wavelength[0]
    minimum = (
        np.round(
            np.min(original_lsf_wavelength * grating_dispersion) / data_spacing
        )
        * data_spacing
    )
    one_sided = np.arange(minimum, 0, data_spacing)
    if len(one_sided) % 2 != 0:
        one_sided = one_sided[1:]
    lsf_wavelength = np.append(
        np.append(one_sided, np.array([0])),
        -one_sided[::-1],
    )
    interpolated = np.interp(
        lsf_wavelength,
        original_lsf_wavelength * grating_dispersion,
        original_lsf / np.sum(original_lsf),
    )
    normalized = interpolated / np.sum(interpolated)
    if len(data_wavelength) < len(normalized):
        normalized = np.delete(normalized, np.where(normalized == 0))
        normalized = np.insert(normalized, 0, 0)
        normalized = np.append(normalized, 0)
    return normalized


make_kernel = gaussian_kernel
ready_stis_lsf = interpolate_stis_lsf
