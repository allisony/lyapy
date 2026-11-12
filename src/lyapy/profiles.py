"""Intrinsic spectral-line profiles used by Lyman-alpha models."""

import numpy as np


LYA_REST_WAVELENGTH = 1215.67
SPEED_OF_LIGHT_KM_S = 2.99792458e5


def gaussian_flux(amplitude, sigma):
    """Return the wavelength-integrated flux of a Gaussian profile."""
    return np.sqrt(2.0 * np.pi) * sigma * amplitude


def lya_intrinsic_profile(
    wavelength,
    narrow_velocity,
    narrow_amplitude,
    narrow_fwhm,
    broad_velocity=2.0,
    broad_amplitude=2.0,
    broad_fwhm=2.0,
    *,
    return_flux=False,
    single_component=False,
    return_individual_components=False,
    line_center=LYA_REST_WAVELENGTH,
):
    """Evaluate the legacy one- or two-Gaussian intrinsic Ly-alpha profile.

    Velocities and full widths at half maximum are in km/s. Amplitudes are
    flux densities, wavelengths and ``line_center`` are in Angstroms, and a
    requested integrated flux is therefore in flux-density Angstroms.
    """
    wavelength = np.asarray(wavelength)
    narrow_center = line_center * (
        1.0 + narrow_velocity / SPEED_OF_LIGHT_KM_S
    )
    narrow_sigma = (
        line_center * (narrow_fwhm / SPEED_OF_LIGHT_KM_S) / 2.3548
    )
    narrow = narrow_amplitude * np.exp(
        -0.5 * ((wavelength - narrow_center) / narrow_sigma) ** 2
    )

    broad_center = line_center * (
        1.0 + broad_velocity / SPEED_OF_LIGHT_KM_S
    )
    broad_sigma = line_center * (broad_fwhm / SPEED_OF_LIGHT_KM_S) / 2.3548
    broad = broad_amplitude * np.exp(
        -0.5 * ((wavelength - broad_center) / broad_sigma) ** 2
    )

    narrow_flux = gaussian_flux(narrow_amplitude, narrow_sigma)
    broad_flux = gaussian_flux(broad_amplitude, broad_sigma)
    if single_component:
        return (narrow, narrow_flux) if return_flux else narrow
    if return_individual_components:
        if return_flux:
            return narrow, broad, narrow_flux + broad_flux
        return narrow, broad
    total = narrow + broad
    return (total, narrow_flux + broad_flux) if return_flux else total


def lya_intrinsic_profile_func(
    x,
    vs_n,
    am_n,
    fw_n,
    vs_b=2.0,
    am_b=2.0,
    fw_b=2.0,
    return_flux=False,
    single_component_flux=False,
    return_individual_components=False,
    line_center=LYA_REST_WAVELENGTH,
):
    """Compatibility spelling and argument names for the historical API."""
    return lya_intrinsic_profile(
        x,
        vs_n,
        am_n,
        fw_n,
        vs_b,
        am_b,
        fw_b,
        return_flux=return_flux,
        single_component=single_component_flux,
        return_individual_components=return_individual_components,
        line_center=line_center,
    )
