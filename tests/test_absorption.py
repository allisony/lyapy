import numpy as np
import pytest

from lyapy import lyapy
from lyapy.absorption import LINE_DATA, tau_profile, total_tau_profile
from lyapy.atomic_data import ATOMIC_TRANSITIONS


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


def test_absorption_uses_catalog_transition_data():
    transition = ATOMIC_TRANSITIONS["n_v_1238"]
    from_key = tau_profile(14.0, 4.0, 12.0, "n_v_1238")
    from_transition = tau_profile(14.0, 4.0, 12.0, transition)

    assert LINE_DATA["n_v_1238"] == transition.absorption_parameters()
    np.testing.assert_array_equal(from_key[0], from_transition[0])
    np.testing.assert_array_equal(from_key[1], from_transition[1])


def test_legacy_h1_constants_remain_separate_from_morton_catalog_values():
    assert LINE_DATA["h1"] == (1215.67, 0.4161, 6.26e8)
    assert LINE_DATA["h_i_1215"] == (1215.6700, 0.4164, 6.265e8)


def test_mg_ii_has_correct_canonical_names_and_legacy_aliases():
    assert LINE_DATA["mg_ii_k"][0] == 2796.3543
    assert LINE_DATA["mg_ii_h"][0] == 2803.5315

    with pytest.warns(DeprecationWarning, match="historically reversed"):
        legacy_k = tau_profile(14.0, 0.0, 3.0, "mg2_h")
    canonical_k = tau_profile(14.0, 0.0, 3.0, "mg_ii_k")
    np.testing.assert_array_equal(legacy_k[0], canonical_k[0])
    np.testing.assert_array_equal(legacy_k[1], canonical_k[1])
