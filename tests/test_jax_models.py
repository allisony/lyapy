import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.special import voigt_profile

from lyapy.jax_models import (
    _interpolate_harris_tables,
    evaluate_voigt_lya_model_jax,
    line_optical_depth_jax,
    normalized_voigt_profile_jax,
    transition_optical_depth_jax,
)
from lyapy import voigt as legacy_voigt
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


@pytest.fixture(scope="module")
def model_inputs():
    reference = evaluate_gj411_legacy_case()
    wave = np.asarray(reference["wave"], dtype=np.float64)
    kernel = np.asarray(reference["lsf_kernel"], dtype=np.float64)
    parameters = np.asarray(PARAMETERS, dtype=np.float64)
    return reference, jnp.asarray(wave), jnp.asarray(kernel), jnp.asarray(parameters)


def test_jax_model_matches_numpy_baseline(model_inputs):
    reference, wave, kernel, parameters = model_inputs
    result = evaluate_voigt_lya_model_jax(wave, parameters, kernel)

    np.testing.assert_allclose(
        result.stellar_voigt,
        reference["stellar_voigt"],
        rtol=4e-5,
        atol=1e-24,
    )
    attenuation_error = np.max(
        np.abs(result.unconvolved.attenuation - reference["ism_total"])
    )
    assert attenuation_error < 1e-4

    model_error = np.max(
        np.abs(result.convolved.observed - reference["model_convolved"])
    ) / np.max(np.abs(reference["model_convolved"]))
    assert model_error < 6e-5


def test_direct_harris_table_interpolation_matches_binary_search():
    u = jnp.asarray(
        [0.0, 0.0037, 1.2345, 7.8912, 19.985, 19.99, 25.0]
    )
    expected_h1 = jnp.interp(u, legacy_voigt.uvals, legacy_voigt.H1)
    expected_h3 = jnp.interp(u, legacy_voigt.uvals, legacy_voigt.H3)
    actual_h1, actual_h3 = _interpolate_harris_tables(u)

    np.testing.assert_allclose(actual_h1, expected_h1, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(actual_h3, expected_h3, rtol=2e-14, atol=1e-15)

    test_u = jnp.asarray([0.0037, 1.2345, 7.8912, 19.985])
    actual_gradient = jax.grad(
        lambda values: sum(
            jnp.sum(table)
            for table in _interpolate_harris_tables(values)
        )
    )(test_u)
    expected_gradient = jax.grad(
        lambda values: jnp.sum(
            jnp.interp(values, legacy_voigt.uvals, legacy_voigt.H1)
            + jnp.interp(values, legacy_voigt.uvals, legacy_voigt.H3)
        )
    )(test_u)
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=2e-12,
        atol=2e-12,
    )


def test_normalized_jax_voigt_matches_scipy_probability_density():
    wavelength = np.linspace(-2.0, 2.0, 20_001)
    gaussian_fwhm = 0.24
    lorentz_fwhm = 0.05
    sigma = gaussian_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = 0.5 * lorentz_fwhm

    actual = normalized_voigt_profile_jax(
        wavelength,
        0.0,
        lorentz_fwhm,
        gaussian_fwhm,
    )
    expected = voigt_profile(wavelength, sigma, gamma)

    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=1e-14)


def test_jax_optical_depth_accepts_atomic_catalog_key():
    wavelength = jnp.linspace(1238.3, 1239.3, 101)
    from_catalog = transition_optical_depth_jax(
        wavelength,
        14.0,
        4.0,
        12.0,
        "n_v_1238",
    )
    explicit = line_optical_depth_jax(
        wavelength,
        14.0,
        4.0,
        12.0,
        1238.821,
        1.560e-1,
        3.391e8,
    )

    np.testing.assert_array_equal(from_catalog, explicit)


def test_jax_model_jit_matches_eager_evaluation(model_inputs):
    reference, wave, kernel, parameters = model_inputs
    eager = evaluate_voigt_lya_model_jax(wave, parameters, kernel)
    compiled_model = jax.jit(evaluate_voigt_lya_model_jax)
    compiled = compiled_model(wave, parameters, kernel)

    np.testing.assert_allclose(
        compiled.convolved.observed,
        eager.convolved.observed,
        rtol=1e-13,
        atol=1e-24,
    )


def test_jax_model_has_finite_parameter_gradients(model_inputs):
    _, wave, kernel, parameters = model_inputs

    def scaled_integrated_flux(parameter_vector):
        result = evaluate_voigt_lya_model_jax(wave, parameter_vector, kernel)
        return jnp.trapezoid(result.convolved.observed, wave) / 1e-12

    gradient = jax.grad(scaled_integrated_flux)(parameters)

    assert gradient.shape == parameters.shape
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize(
    "parameter_index, step",
    [
        (0, 1e-3),
        (1, 1e-5),
        (4, 1e-5),
        (6, 3e-5),
        (11, 1e-5),
    ],
)
def test_jax_gradient_matches_central_difference(model_inputs, parameter_index, step):
    _, wave, kernel, parameters = model_inputs

    def scaled_integrated_flux(parameter_vector):
        result = evaluate_voigt_lya_model_jax(wave, parameter_vector, kernel)
        return jnp.trapezoid(result.convolved.observed, wave) / 1e-12

    automatic = jax.grad(scaled_integrated_flux)(parameters)[parameter_index]
    offset = np.zeros(len(parameters))
    offset[parameter_index] = step
    finite_difference = (
        scaled_integrated_flux(parameters + offset)
        - scaled_integrated_flux(parameters - offset)
    ) / (2.0 * step)

    np.testing.assert_allclose(automatic, finite_difference, rtol=2e-4, atol=1e-7)
