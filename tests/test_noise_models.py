from functools import partial
from typing import NamedTuple

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
pytest.importorskip("numpyro")
from numpyro.infer.util import log_density

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.comparison import ModelHypothesis
from lyapy.inference.dynesty_backend import build_dynesty_functions
from lyapy.inference.noise import (
    GaussianNoiseModel,
    StudentTNoiseModel,
    augment_configuration_with_noise,
)
from lyapy.inference.numpyro_backend import numpyro_spectrum_model, run_nuts


class Convolved(NamedTuple):
    observed: object


class Result(NamedTuple):
    convolved: Convolved


def constant_forward_model(wavelength, parameters, convolution_kernel):
    del convolution_kernel
    return Result(Convolved(jnp.full_like(wavelength, parameters[0])))


def make_problem():
    data = SpectrumData(
        wavelength=np.array([1.0, 2.0, 3.0, 4.0]),
        flux=np.array([1.0, 1.2, 0.8, 1.1]),
        error=np.array([0.1, 0.01, 0.1, 0.1]),
        convolution_kernel=np.array([1.0]),
    )
    configuration = InferenceConfiguration(
        base_parameters=(1.0,),
        free_parameters=(UniformParameter("level", 0, 0.0, 2.0),),
        parameter_names=("level",),
    )
    return data, configuration


def test_gaussian_noise_combines_scale_and_jitter_in_quadrature():
    model = GaussianNoiseModel(
        ln_error_scale_index=1,
        log10_jitter_index=2,
    )
    parameters = np.array([0.0, np.log(2.0), np.log10(0.3)])
    error = np.array([0.1, 0.2])
    expected = np.sqrt((2.0 * error) ** 2 + 0.3**2)

    np.testing.assert_allclose(
        model.effective_error_numpy(error, parameters),
        expected,
    )
    np.testing.assert_allclose(
        model.effective_error_jax(error, jnp.asarray(parameters)),
        expected,
    )


def test_student_t_is_robust_to_one_large_outlier():
    observed = np.array([0.0, 0.0, 10.0])
    predicted = np.zeros(3)
    error = np.ones(3)
    parameters = np.array([0.0])
    gaussian = GaussianNoiseModel()
    student_t = StudentTNoiseModel(degrees_of_freedom=4.0)

    assert student_t(
        observed,
        predicted,
        error,
        parameters,
    ) > gaussian(observed, predicted, error, parameters)
    np.testing.assert_allclose(
        student_t.numpy_log_likelihood(observed, predicted, error, parameters),
        student_t.jax_log_likelihood(
            observed,
            predicted,
            error,
            jnp.asarray(parameters),
        ),
        rtol=1e-13,
    )


def test_configuration_augmentation_appends_explicit_noise_parameters():
    _, configuration = make_problem()
    augmented, likelihood = augment_configuration_with_noise(configuration)

    assert augmented.parameter_names == (
        "level",
        "ln_error_scale",
        "log10_jitter",
    )
    assert likelihood.ln_error_scale_index == 1
    assert likelihood.log10_jitter_index == 2
    assert tuple(
        parameter.name for parameter in augmented.free_parameters
    ) == augmented.parameter_names


def test_dynesty_and_numpyro_agree_for_parameterized_noise_model():
    data, configuration = make_problem()
    augmented, likelihood = augment_configuration_with_noise(
        configuration,
        log10_jitter_bounds=(-3.0, 0.0),
        base_log10_jitter=-2.0,
    )
    free_values = np.asarray(augmented.base_parameters)
    dynesty_log_likelihood, _ = build_dynesty_functions(
        data,
        augmented,
        forward_model=constant_forward_model,
        likelihood_function=likelihood,
    )
    values = {
        parameter.name: free_values[position]
        for position, parameter in enumerate(augmented.free_parameters)
    }
    model = partial(
        numpyro_spectrum_model,
        forward_model=constant_forward_model,
        likelihood_function=likelihood,
    )
    numpyro_log_density, _ = log_density(
        model,
        (data, augmented),
        {},
        values,
    )
    uniform_log_prior = -sum(
        np.log(parameter.upper - parameter.lower)
        for parameter in augmented.free_parameters
    )

    np.testing.assert_allclose(
        numpyro_log_density,
        dynesty_log_likelihood(free_values) + uniform_log_prior,
        rtol=1e-13,
    )


def test_noise_model_serialization_records_distribution_and_indices():
    _, configuration = make_problem()
    augmented, likelihood = augment_configuration_with_noise(
        configuration,
        distribution="student_t",
        degrees_of_freedom=5.0,
    )
    report = ModelHypothesis(
        "robust",
        augmented,
        forward_model=constant_forward_model,
        likelihood_function=likelihood,
    ).as_dict()

    assert report["likelihood_specification"]["distribution"] == "StudentT"
    assert report["likelihood_specification"]["degrees_of_freedom"] == 5.0
    assert report["likelihood_specification"]["ln_error_scale_index"] == 1


def test_numpyro_samples_noise_parameters():
    data, configuration = make_problem()
    augmented, likelihood = augment_configuration_with_noise(
        configuration,
        include_jitter=False,
    )
    sampler = run_nuts(
        data,
        augmented,
        jax.random.key(31),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        forward_model=constant_forward_model,
        likelihood_function=likelihood,
    )
    samples = sampler.get_samples()

    assert samples["ln_error_scale"].shape == (10,)
    assert samples["model_flux"].shape == (10, data.wavelength.size)
    assert np.all(np.isfinite(samples["model_flux"]))
