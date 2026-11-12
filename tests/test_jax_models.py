import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.jax_models import evaluate_voigt_lya_model_jax
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
