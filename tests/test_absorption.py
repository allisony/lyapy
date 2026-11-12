import numpy as np

from lyapy import lyapy


def test_tau_profile_is_finite_nonnegative_and_symmetric_at_zero_velocity():
    wave, optical_depth = lyapy.tau_profile(
        ncols=18.0,
        vshifts=0.0,
        vdop=12.0,
        which_line="h1",
    )

    assert wave.shape == optical_depth.shape
    assert np.all(np.isfinite(optical_depth))
    assert np.all(optical_depth >= 0.0)
    # The legacy interpolation grid introduces relative roundoff at ~1e-11.
    np.testing.assert_allclose(optical_depth, optical_depth[::-1], rtol=2e-11)


def test_total_attenuation_is_a_transmission_fraction():
    wave = np.linspace(1214.5, 1216.55, 801)
    attenuation = lyapy.total_tau_profile_func(
        wave,
        h1_col=18.0,
        h1_b=12.0,
        h1_vel=3.0,
        d2h=1.5e-5,
    )

    assert np.all(np.isfinite(attenuation))
    assert np.all(attenuation >= 0.0)
    assert np.all(attenuation <= 1.0)
    assert np.min(attenuation) < 1.0e-6
