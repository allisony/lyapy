import numpy as np

from lyapy import lyapy
from lyapy.absorption import tau_profile, total_tau_profile


def test_tau_profile_is_finite_nonnegative_and_symmetric_at_zero_velocity():
    wave, optical_depth = tau_profile(
        log_column_density=18.0,
        velocity_shift=0.0,
        doppler_b=12.0,
        line="h1",
    )

    assert wave.shape == optical_depth.shape
    assert np.all(np.isfinite(optical_depth))
    assert np.all(optical_depth >= 0.0)
    # The legacy interpolation grid introduces relative roundoff at ~1e-11.
    np.testing.assert_allclose(optical_depth, optical_depth[::-1], rtol=2e-11)


def test_total_attenuation_is_a_transmission_fraction():
    wave = np.linspace(1214.5, 1216.55, 801)
    attenuation = total_tau_profile(
        wave,
        h1_column=18.0,
        h1_doppler_b=12.0,
        h1_velocity=3.0,
        deuterium_to_hydrogen=1.5e-5,
    )

    assert np.all(np.isfinite(attenuation))
    assert np.all(attenuation >= 0.0)
    assert np.all(attenuation <= 1.0)
    assert np.min(attenuation) < 1.0e-6


def test_legacy_absorption_wrapper_matches_public_api():
    wave = np.linspace(1214.5, 1216.55, 101)
    legacy = lyapy.total_tau_profile_func(wave, 18.0, 12.0, 3.0)
    public = total_tau_profile(wave, 18.0, 12.0, 3.0)

    np.testing.assert_array_equal(legacy, public)
