import numpy as np

from lyapy import lyapy


def test_gaussian_flux_matches_numerical_integral():
    amplitude = 3.2
    sigma = 0.17
    grid = np.linspace(-8.0 * sigma, 8.0 * sigma, 20_001)
    profile = amplitude * np.exp(-0.5 * (grid / sigma) ** 2)

    expected = lyapy.gaussian_flux(amplitude, sigma)
    actual = np.trapezoid(profile, grid)

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_intrinsic_profile_components_sum_to_total():
    wave = np.linspace(1212.0, 1219.0, 20_001)
    narrow, broad = lyapy.lya_intrinsic_profile_func(
        wave,
        vs_n=-8.0,
        am_n=2.0,
        fw_n=60.0,
        vs_b=5.0,
        am_b=0.7,
        fw_b=140.0,
        return_individual_components=True,
    )
    total = lyapy.lya_intrinsic_profile_func(
        wave,
        vs_n=-8.0,
        am_n=2.0,
        fw_n=60.0,
        vs_b=5.0,
        am_b=0.7,
        fw_b=140.0,
    )

    np.testing.assert_allclose(total, narrow + broad, rtol=0.0, atol=0.0)


def test_intrinsic_profile_reported_flux_matches_integral():
    wave = np.linspace(1212.0, 1219.0, 20_001)
    profile, reported_flux = lyapy.lya_intrinsic_profile_func(
        wave,
        vs_n=-8.0,
        am_n=2.0,
        fw_n=60.0,
        vs_b=5.0,
        am_b=0.7,
        fw_b=140.0,
        return_flux=True,
    )

    np.testing.assert_allclose(
        np.trapezoid(profile, wave), reported_flux, rtol=1e-12
    )
