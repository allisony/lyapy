import numpy as np

from lyapy import lyapy
from lyapy.profiles import gaussian_flux, lya_intrinsic_profile


def test_gaussian_flux_matches_numerical_integral():
    amplitude = 3.2
    sigma = 0.17
    grid = np.linspace(-8.0 * sigma, 8.0 * sigma, 20_001)
    profile = amplitude * np.exp(-0.5 * (grid / sigma) ** 2)

    expected = gaussian_flux(amplitude, sigma)
    actual = np.trapezoid(profile, grid)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_intrinsic_profile_components_sum_to_total():
    wave = np.linspace(1212.0, 1219.0, 20_001)
    narrow, broad = lya_intrinsic_profile(
        wave,
        narrow_velocity=-8.0,
        narrow_amplitude=2.0,
        narrow_fwhm=60.0,
        broad_velocity=5.0,
        broad_amplitude=0.7,
        broad_fwhm=140.0,
        return_individual_components=True,
    )
    total = lya_intrinsic_profile(
        wave,
        narrow_velocity=-8.0,
        narrow_amplitude=2.0,
        narrow_fwhm=60.0,
        broad_velocity=5.0,
        broad_amplitude=0.7,
        broad_fwhm=140.0,
    )

    np.testing.assert_allclose(total, narrow + broad, rtol=0.0, atol=0.0)


def test_intrinsic_profile_reported_flux_matches_integral():
    wave = np.linspace(1212.0, 1219.0, 20_001)
    profile, reported_flux = lya_intrinsic_profile(
        wave,
        narrow_velocity=-8.0,
        narrow_amplitude=2.0,
        narrow_fwhm=60.0,
        broad_velocity=5.0,
        broad_amplitude=0.7,
        broad_fwhm=140.0,
        return_flux=True,
    )

    np.testing.assert_allclose(
        np.trapezoid(profile, wave), reported_flux, rtol=1e-12
    )


def test_legacy_profile_wrapper_matches_public_api():
    wave = np.linspace(1214.5, 1216.5, 101)
    arguments = (-8.0, 2.0, 60.0, 5.0, 0.7, 140.0)

    np.testing.assert_array_equal(
        lyapy.lya_intrinsic_profile_func(wave, *arguments),
        lya_intrinsic_profile(wave, *arguments),
    )
