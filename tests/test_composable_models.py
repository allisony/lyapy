import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.dynesty_backend import build_dynesty_functions
from lyapy.jax_components import (
    CompositeEmissionModel,
    GaussianEmissionComponent,
    GaussianOpticalDepthReversal,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import (
    build_high_resolution_operator,
    detector_bin_edges,
)


def gaussian_lsf():
    offsets = np.linspace(-0.15, 0.15, 301)
    profile = np.exp(-0.5 * (offsets / 0.025) ** 2)
    return offsets, profile


def test_flux_parameter_normalizes_gaussian_and_voigt_profiles():
    wavelength = jnp.linspace(1213.0, 1218.0, 20_001)
    parameters = jnp.array([-25.0, -11.2, 9.0, 85.0])
    gaussian = CompositeEmissionModel(
        components=(GaussianEmissionComponent(1215.67, 0, 3),),
        log_total_flux_index=1,
    )
    voigt = CompositeEmissionModel(
        components=(VoigtEmissionComponent(1215.67, 0, 3, 2),),
        log_total_flux_index=1,
    )

    gaussian_result = gaussian.evaluate(wavelength, parameters)
    voigt_result = voigt.evaluate(wavelength, parameters)

    np.testing.assert_allclose(
        np.trapezoid(gaussian_result.intrinsic, wavelength),
        10.0 ** parameters[1],
        rtol=2e-13,
    )
    np.testing.assert_allclose(gaussian_result.normalization_integral, 1.0)
    np.testing.assert_allclose(voigt_result.normalization_integral, 1.0)
    assert np.trapezoid(voigt_result.intrinsic, wavelength) < 10.0 ** parameters[1]


def test_multiple_components_use_relative_weights_and_one_total_flux():
    wavelength = jnp.linspace(1213.0, 1218.0, 20_001)
    parameters = jnp.array([-20.0, -11.0, 55.0, 15.0, 140.0, np.log10(0.4)])
    model = CompositeEmissionModel(
        components=(
            GaussianEmissionComponent(1215.67, 0, 2),
            GaussianEmissionComponent(1215.67, 3, 4, log_weight_index=5),
        ),
        log_total_flux_index=1,
    )

    result = model.evaluate(wavelength, parameters)

    np.testing.assert_allclose(result.component_weights, [1.0, 0.4])
    np.testing.assert_allclose(
        np.trapezoid(result.intrinsic, wavelength),
        1e-11,
        rtol=2e-13,
    )


def test_reversal_changes_shape_without_changing_total_flux():
    wavelength = jnp.linspace(1214.0, 1217.0, 12_001)
    parameters = jnp.array([-20.0, -11.0, 90.0, -20.0, 35.0, 1.4])
    component = GaussianEmissionComponent(1215.67, 0, 2)
    unreversed = CompositeEmissionModel((component,), 1)
    reversed_model = CompositeEmissionModel(
        (component,),
        1,
        reversal=GaussianOpticalDepthReversal(1215.67, 3, 4, 5),
    )

    plain = unreversed.evaluate(wavelength, parameters)
    reversed_result = reversed_model.evaluate(wavelength, parameters)
    center_index = int(np.argmin(np.abs(wavelength - 1215.67 * (1 - 20 / 2.99792458e5))))

    assert reversed_result.intrinsic[center_index] < plain.intrinsic[center_index]
    np.testing.assert_allclose(
        np.trapezoid(reversed_result.intrinsic, wavelength),
        1e-11,
        rtol=2e-13,
    )


def test_detector_edges_and_operator_preserve_constant_flux_density():
    centers = np.linspace(1214.5, 1216.5, 101)
    offsets, lsf = gaussian_lsf()
    edges = detector_bin_edges(centers)
    operator = build_high_resolution_operator(
        centers,
        offsets,
        lsf,
        oversample=8,
        model_padding=0.5,
    )

    np.testing.assert_allclose(centers, 0.5 * (edges[:-1] + edges[1:]))
    _, detector_flux = operator.apply(
        jnp.full(operator.wavelength.shape, 3.2)
    )
    np.testing.assert_allclose(detector_flux, 3.2, rtol=0.0, atol=2e-14)


def make_binned_gaussian(oversample):
    centers = np.linspace(1214.5, 1216.5, 101)
    offsets, lsf = gaussian_lsf()
    operator = build_high_resolution_operator(
        centers,
        offsets,
        lsf,
        oversample=oversample,
        model_padding=0.5,
    )
    emission = CompositeEmissionModel(
        components=(GaussianEmissionComponent(1215.67, 0, 2),),
        log_total_flux_index=1,
    )
    return centers, operator, ComposableBinnedModel(emission, operator)


def test_high_resolution_prediction_converges_and_conserves_flux():
    parameters = jnp.array([-15.0, -11.0, 22.0])
    predictions = []
    for oversample in (4, 8, 16):
        centers, operator, model = make_binned_gaussian(oversample)
        result = model(centers, parameters, np.array([1.0]))
        predictions.append(np.asarray(result.convolved.observed))
        detector_widths = np.diff(operator.detector_edges)
        np.testing.assert_allclose(
            np.sum(result.convolved.observed * detector_widths),
            1e-11,
            rtol=2e-5,
        )

    error_4_to_16 = np.max(np.abs(predictions[0] - predictions[2]))
    error_8_to_16 = np.max(np.abs(predictions[1] - predictions[2]))
    assert error_8_to_16 < error_4_to_16
    assert error_8_to_16 / np.max(predictions[2]) < 2e-3


def test_voigt_reversal_prediction_is_stable_to_model_padding():
    centers = np.linspace(1214.5, 1216.5, 101)
    offsets, lsf = gaussian_lsf()
    parameters = jnp.array([-20.0, -11.0, 9.0, 85.0, -20.0, 35.0, 1.4])
    emission = CompositeEmissionModel(
        components=(VoigtEmissionComponent(1215.67, 0, 3, 2),),
        log_total_flux_index=1,
        reversal=GaussianOpticalDepthReversal(1215.67, 4, 5, 6),
    )
    predictions = []
    for padding in (0.25, 2.0):
        operator = build_high_resolution_operator(
            centers,
            offsets,
            lsf,
            oversample=16,
            model_padding=padding,
        )
        model = ComposableBinnedModel(emission, operator)
        predictions.append(
            np.asarray(
                model(centers, parameters, np.array([1.0])).convolved.observed
            )
        )

    relative_difference = np.max(np.abs(predictions[0] - predictions[1])) / np.max(
        predictions[1]
    )
    assert relative_difference < 1e-6


def test_convolving_before_detector_binning_differs_from_coarse_order():
    parameters = jnp.array([0.0, -11.0, 5.0])
    centers, operator, model = make_binned_gaussian(16)
    result = model(centers, parameters, np.array([1.0]))
    intrinsic_high = result.unconvolved.intrinsic
    binned_first = np.asarray(operator.binning_matrix) @ np.asarray(intrinsic_high)
    offsets, lsf = gaussian_lsf()
    spacing = centers[1] - centers[0]
    radius = int(np.ceil(max(abs(offsets)) / spacing))
    coarse_offsets = spacing * np.arange(-radius, radius + 1)
    coarse_kernel = np.interp(coarse_offsets, offsets, lsf, left=0.0, right=0.0)
    coarse_kernel /= np.sum(coarse_kernel)
    wrong_order = np.convolve(binned_first, coarse_kernel, mode="same")

    relative_difference = np.max(
        np.abs(result.convolved.observed - wrong_order)
    ) / np.max(result.convolved.observed)
    assert relative_difference > 0.005


def test_composable_model_is_differentiable_and_works_with_dynesty_adapter():
    centers, _, model = make_binned_gaussian(8)
    truth = jnp.array([-15.0, -11.0, 22.0])
    prediction = np.asarray(model(centers, truth, np.array([1.0])).convolved.observed)
    data = SpectrumData(
        centers,
        prediction,
        np.full(centers.shape, 5e-14),
        np.array([1.0]),
    )
    configuration = InferenceConfiguration(
        base_parameters=tuple(truth),
        free_parameters=(
            UniformParameter("velocity", 0, -40.0, 20.0),
            UniformParameter("log_total_flux", 1, -12.0, -10.0),
            UniformParameter("fwhm", 2, 5.0, 80.0),
        ),
        parameter_names=("velocity", "log_total_flux", "fwhm"),
    )
    log_likelihood, _ = build_dynesty_functions(
        data,
        configuration,
        forward_model=model,
    )

    gradient = jax.grad(
        lambda values: jnp.sum(
            model(centers, values, np.array([1.0])).convolved.observed
        )
        / 1e-11
    )(truth)

    assert np.all(np.isfinite(gradient))
    assert log_likelihood(truth) > log_likelihood(truth + jnp.array([5.0, 0.1, 5.0]))
