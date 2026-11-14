import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference import (
    CII_SPECTRUM_NAME,
    LYA_SPECTRUM_NAME,
    LyaCiiModelDefaults,
    build_lya_cii_model_bundle,
    prepare_e140m_lya_cii_data,
)
from lyapy.inference.common import SpectrumData
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.observations import GJ729_E140M_TARGET


def gaussian_operator(wavelength):
    offsets = np.linspace(-0.15, 0.15, 301)
    profile = np.exp(-0.5 * (offsets / 0.025) ** 2)
    return build_high_resolution_operator(
        wavelength,
        offsets,
        profile,
        oversample=8,
        model_padding=2.0,
    )


def make_bundle():
    lya_wavelength = np.linspace(1214.5, 1216.8, 116)
    cii_wavelength = np.linspace(1333.75, 1336.45, 181)
    defaults = LyaCiiModelDefaults(
        stellar_velocity=-11.0,
        stellar_velocity_bounds=(-36.0, 14.0),
        cii_stellar_velocity=-11.0,
        cii_stellar_velocity_bounds=(-36.0, 14.0),
        lya_log_total_flux=-11.8,
        cii_log_total_flux=-13.0,
        cii_1335a_log_weight=-0.5,
        cii_1335b_log_weight=0.2,
    )
    bundle = build_lya_cii_model_bundle(
        gaussian_operator(lya_wavelength),
        gaussian_operator(cii_wavelength),
        defaults,
    )
    return lya_wavelength, cii_wavelength, bundle


def test_lya_and_cii_share_only_the_ism_velocity():
    _, _, bundle = make_bundle()
    names = bundle.configuration.parameter_names
    index = names.index
    h1 = bundle.lya_model.absorption_model.components[0]
    cii = bundle.cii_model.absorption_model.components

    assert h1.h1_velocity_index == index("ism_velocity")
    assert all(item.velocity_index == index("ism_velocity") for item in cii)
    assert h1.log_h1_column_index == index("h1_log_column")
    assert cii[0].log_column_index == index("cii_log_column")
    assert cii[1].log_column_index == index("cii_star_log_column")
    assert cii[2].log_column_index == index("cii_star_log_column")
    assert h1.h1_doppler_b_index != cii[0].doppler_b_index
    assert all(
        component.velocity_index == index("cii_stellar_velocity")
        for component in bundle.cii_model.emission_model.components
    )
    assert all(
        component.velocity_index != index("stellar_velocity")
        for component in bundle.cii_model.emission_model.components
    )


def test_cii_multiplet_has_free_emission_ratios_and_separate_absorption():
    _, cii_wavelength, bundle = make_bundle()
    parameters = np.asarray(bundle.configuration.base_parameters)
    result = bundle.cii_model(
        cii_wavelength,
        jnp.asarray(parameters),
        np.ones(1),
    )
    weak_absorption = parameters.copy()
    weak_absorption[
        bundle.configuration.parameter_names.index("cii_log_column")
    ] = 11.0
    weak_absorption[
        bundle.configuration.parameter_names.index("cii_star_log_column")
    ] = 10.0
    unobscured = bundle.cii_model(
        cii_wavelength,
        jnp.asarray(weak_absorption),
        np.ones(1),
    )

    np.testing.assert_allclose(
        result.unconvolved.component_weights,
        [1.0, 10.0**-0.5, 10.0**0.2],
    )
    assert np.max(
        np.abs(
            np.asarray(result.convolved.observed)
            - np.asarray(unobscured.convolved.observed)
        )
    ) > 0.0


def test_bundle_builds_a_standard_joint_inference_problem():
    lya_wavelength, cii_wavelength, bundle = make_bundle()
    parameters = jnp.asarray(bundle.configuration.base_parameters)
    empty_lya = SpectrumData(
        lya_wavelength,
        np.zeros(lya_wavelength.size),
        np.ones(lya_wavelength.size),
        np.ones(1),
    )
    empty_cii = SpectrumData(
        cii_wavelength,
        np.zeros(cii_wavelength.size),
        np.ones(cii_wavelength.size),
        np.ones(1),
    )
    empty_problem = bundle.problem(empty_lya, empty_cii)
    predictions = empty_problem.forward_model.predict(
        empty_problem.data,
        parameters,
    )
    data = []
    for wavelength, prediction in zip(
        (lya_wavelength, cii_wavelength),
        predictions,
    ):
        data.append(
            SpectrumData(
                wavelength,
                np.asarray(prediction),
                np.full(wavelength.size, 1e-14),
                np.ones(1),
            )
        )
    problem = bundle.problem(*data, name="synthetic Ly-alpha plus C II")

    assert problem.data.names == (LYA_SPECTRUM_NAME, CII_SPECTRUM_NAME)
    assert problem.forward_model.names == problem.data.names
    assert all(np.all(np.isfinite(value)) for value in predictions)


def test_packaged_e140m_target_exposes_native_lya_and_cii_windows():
    lya, cii = prepare_e140m_lya_cii_data(GJ729_E140M_TARGET)

    assert lya.wavelength[0] >= GJ729_E140M_TARGET.fit_window[0]
    assert lya.wavelength[-1] <= GJ729_E140M_TARGET.fit_window[1]
    assert cii.wavelength[0] >= 1333.75
    assert cii.wavelength[-1] <= 1336.45
    assert not np.all(lya.likelihood_mask)
    assert np.all(cii.likelihood_mask)
    assert lya.metadata["continuum_subtracted"] is False
    assert cii.metadata["continuum_subtracted"] is False
    assert lya.metadata["uncertainty_scale"] == pytest.approx(np.sqrt(2.0))
