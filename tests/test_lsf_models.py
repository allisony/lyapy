import math

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.inference.lsf import augment_configuration_with_lsf_width
from lyapy.jax_components import CompositeEmissionModel, GaussianEmissionComponent
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator


def make_model():
    centers = np.linspace(1215.2, 1216.1, 91)
    offsets = np.linspace(-0.12, 0.12, 241)
    profile = np.exp(-0.5 * (offsets / 0.018) ** 2)
    operator = build_high_resolution_operator(
        centers,
        offsets,
        profile,
        oversample=8,
        model_padding=0.3,
    )
    configuration = InferenceConfiguration(
        base_parameters=(0.0, -11.0, 35.0),
        free_parameters=(
            UniformParameter("velocity", 0, -20.0, 20.0),
            UniformParameter("log_total_flux", 1, -12.0, -10.0),
            UniformParameter("fwhm", 2, 10.0, 80.0),
        ),
        parameter_names=("velocity", "log_total_flux", "fwhm"),
    )
    emission = CompositeEmissionModel(
        (GaussianEmissionComponent(1215.67, 0, 2),),
        1,
    )
    return centers, configuration, ComposableBinnedModel(emission, operator)


def kernel_variance(operator, parameters):
    kernel = np.asarray(operator.effective_lsf_kernel(parameters))
    offsets = np.asarray(operator.kernel_offsets)
    mean = np.sum(kernel * offsets)
    return np.sum(kernel * (offsets - mean) ** 2)


def test_configuration_augmentation_adds_log_width_scale_and_matched_operator():
    _, configuration, model = make_model()
    augmented, fitted_model = augment_configuration_with_lsf_width(
        configuration,
        model,
    )

    parameter = augmented.free_parameters[-1]
    assert parameter.name == "ln_lsf_width_scale"
    assert parameter.index == 3
    assert parameter.lower == pytest.approx(math.log(0.8))
    assert parameter.upper == pytest.approx(math.log(1.25))
    assert fitted_model.observation_operator.ln_lsf_width_scale_index == 3


def test_unit_width_reproduces_fixed_lsf_detector_prediction():
    centers, configuration, model = make_model()
    augmented, fitted_model = augment_configuration_with_lsf_width(
        configuration,
        model,
    )
    fixed = model(
        centers,
        jnp.asarray(configuration.base_parameters),
        np.array([1.0]),
    ).convolved.observed
    fitted = fitted_model(
        centers,
        jnp.asarray(augmented.base_parameters),
        np.array([1.0]),
    ).convolved.observed

    np.testing.assert_allclose(fitted, fixed, rtol=2e-15, atol=1e-27)


def test_width_scale_broadens_full_lsf_and_preserves_normalization():
    _, configuration, model = make_model()
    augmented, fitted_model = augment_configuration_with_lsf_width(
        configuration,
        model,
    )
    operator = fitted_model.observation_operator
    parameters = np.asarray(augmented.base_parameters)
    narrow = parameters.copy()
    broad = parameters.copy()
    narrow[-1] = math.log(0.85)
    broad[-1] = math.log(1.2)

    narrow_kernel = np.asarray(operator.effective_lsf_kernel(narrow))
    broad_kernel = np.asarray(operator.effective_lsf_kernel(broad))
    np.testing.assert_allclose(np.sum(narrow_kernel), 1.0, atol=2e-15)
    np.testing.assert_allclose(np.sum(broad_kernel), 1.0, atol=2e-15)
    assert kernel_variance(operator, broad) > kernel_variance(operator, narrow)


def test_fitted_lsf_width_is_differentiable_in_detector_space():
    centers, configuration, model = make_model()
    augmented, fitted_model = augment_configuration_with_lsf_width(
        configuration,
        model,
    )
    parameters = jnp.asarray(augmented.base_parameters)

    gradient = jax.grad(
        lambda values: jnp.max(
            fitted_model(centers, values, np.array([1.0])).convolved.observed
        )
        / 1e-11
    )(parameters)

    assert np.all(np.isfinite(gradient))
    assert abs(gradient[-1]) > 0.0


def test_lsf_augmentation_rejects_duplicate_or_nonphysical_parameters():
    _, configuration, model = make_model()
    augmented, fitted_model = augment_configuration_with_lsf_width(
        configuration,
        model,
    )

    with pytest.raises(ValueError, match="already contains"):
        augment_configuration_with_lsf_width(augmented, fitted_model)
    with pytest.raises(ValueError, match="positive and increasing"):
        augment_configuration_with_lsf_width(
            configuration,
            model,
            width_scale_bounds=(1.2, 0.8),
        )
