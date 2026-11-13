import json

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.spectral_lines import (
    SPECTRAL_LINE_GROUPS,
    AtomicTransition,
    SpectralLineGroup,
    gaussian_components_for_line_group,
    gaussian_emission_model_for_line_group,
    get_spectral_line_group,
    voigt_components_for_line_group,
)


def test_catalog_contains_requested_uv_line_groups_and_vacuum_wavelengths():
    assert tuple(SPECTRAL_LINE_GROUPS) == (
        "lya",
        "si_iii_1206",
        "n_v_1238_1242",
        "o_v_1218",
        "c_ii_1334_1335",
        "mg_ii_hk",
    )
    assert [
        transition.rest_wavelength
        for transition in SPECTRAL_LINE_GROUPS["n_v_1238_1242"].transitions
    ] == [1238.821, 1242.804]
    assert [
        transition.rest_wavelength
        for transition in SPECTRAL_LINE_GROUPS["mg_ii_hk"].transitions
    ] == [2796.3543, 2803.5315]
    json.dumps(
        {
            key: group.as_dict()
            for key, group in SPECTRAL_LINE_GROUPS.items()
        }
    )


def test_multiplet_ratios_must_be_explicit_even_when_cataloged():
    group = SPECTRAL_LINE_GROUPS["n_v_1238_1242"]

    with pytest.raises(ValueError, match="multiplet ratios must be explicit"):
        gaussian_components_for_line_group(group, 0, 2)

    components = gaussian_components_for_line_group(
        group,
        0,
        2,
        relative_weights=group.optically_thin_relative_strengths,
    )
    parameters = jnp.asarray([-20.0, -12.0, 55.0])
    np.testing.assert_allclose(
        [component.weight(parameters) for component in components],
        [2.0, 1.0],
    )


def test_tied_multiplet_uses_one_total_flux_and_shared_kinematics():
    group = SPECTRAL_LINE_GROUPS["n_v_1238_1242"]
    model = gaussian_emission_model_for_line_group(
        group,
        velocity_index=0,
        fwhm_index=2,
        log_total_flux_index=1,
        relative_weights=(2.0, 1.0),
    )
    wavelength = jnp.linspace(1235.0, 1246.0, 44_001)
    parameters = jnp.asarray([-20.0, -12.0, 55.0])

    result = model.evaluate(wavelength, parameters)

    np.testing.assert_allclose(result.component_weights, [2.0, 1.0])
    np.testing.assert_allclose(
        np.trapezoid(result.intrinsic, wavelength),
        1e-12,
        rtol=2e-13,
    )
    component_fluxes = np.trapezoid(
        result.total_flux
        * result.component_weights[:, None]
        * result.component_shapes
        / result.normalization_integral,
        wavelength,
        axis=1,
    )
    np.testing.assert_allclose(component_fluxes[0] / component_fluxes[1], 2.0)


def test_free_relative_weights_are_anchored_to_first_transition():
    components = gaussian_components_for_line_group(
        "c_ii_1334_1335",
        velocity_index=0,
        fwhm_index=2,
        log_relative_weight_indices=(3, 4),
    )
    parameters = jnp.asarray([-5.0, -13.0, 40.0, np.log10(0.4), np.log10(0.2)])

    np.testing.assert_allclose(
        [component.weight(parameters) for component in components],
        [1.0, 0.4, 0.2],
    )
    assert all(component.velocity_index == 0 for component in components)
    assert all(component.fwhm_index == 2 for component in components)


def test_singlets_need_no_ratio_and_voigt_multiplets_share_widths():
    singlet = gaussian_components_for_line_group("si_iii_1206", 0, 2)
    assert len(singlet) == 1
    assert singlet[0].rest_wavelength == 1206.500

    doublet = voigt_components_for_line_group(
        "mg_ii_hk",
        velocity_index=0,
        gaussian_fwhm_index=2,
        lorentz_fwhm_index=3,
        relative_weights=(1.0, 1.0),
    )
    assert all(component.velocity_index == 0 for component in doublet)
    assert all(component.gaussian_fwhm_index == 2 for component in doublet)
    assert all(component.lorentz_fwhm_index == 3 for component in doublet)


def test_user_defined_line_groups_use_the_same_constructors():
    custom = SpectralLineGroup(
        "custom",
        "Custom doublet",
        (
            AtomicTransition("custom_a", "X", "X a", 1000.0),
            AtomicTransition("custom_b", "X", "X b", 1001.0),
        ),
    )
    components = gaussian_components_for_line_group(
        custom,
        0,
        2,
        relative_weights=(1.0, 0.5),
    )

    assert get_spectral_line_group(custom) is custom
    assert [component.rest_wavelength for component in components] == [
        1000.0,
        1001.0,
    ]
    with pytest.raises(TypeError):
        SPECTRAL_LINE_GROUPS["custom"] = custom
