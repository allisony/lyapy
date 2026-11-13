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
from lyapy.inference.result_diagnostics import (
    effective_error_samples,
    posterior_predictive_samples,
)
from lyapy.inference.workflow import InferenceProblem, InferenceResult


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


def make_portable_noise_result(distribution="gaussian"):
    data, configuration = make_problem()
    augmented, likelihood = augment_configuration_with_noise(
        configuration,
        log10_jitter_bounds=(-3.0, 0.0),
        base_log10_jitter=-1.0,
        distribution=distribution,
        degrees_of_freedom=5.0,
    )
    problem = InferenceProblem(
        data,
        augmented,
        constant_forward_model,
        likelihood_function=likelihood,
        name=f"{distribution} noise",
    )
    samples = np.array(
        [
            [1.0, np.log(2.0), -1.0],
            [1.1, np.log(2.0), -1.0],
            [0.9, np.log(2.0), -1.0],
        ]
    )
    predictions = np.repeat(samples[:, :1], data.wavelength.size, axis=1)
    return InferenceResult(
        name=problem.name,
        backend="nuts",
        parameter_names=tuple(
            parameter.name for parameter in augmented.free_parameters
        ),
        samples=samples,
        weights=None,
        diagnostics={},
        problem_specification=problem.as_dict(),
        observations={"spectrum": data},
        predictions={"spectrum": predictions},
    )


def test_portable_diagnostics_reconstruct_fitted_scale_and_jitter():
    result = make_portable_noise_result()
    errors = effective_error_samples(result)
    expected = np.sqrt(
        (2.0 * result.observations["spectrum"].error) ** 2 + 0.1**2
    )

    assert errors.shape == (3, 4)
    np.testing.assert_allclose(
        errors,
        np.broadcast_to(expected, errors.shape),
    )
    diagnostics = result.fit_diagnostics(random_seed=59)
    np.testing.assert_allclose(diagnostics.central_effective_error, expected)
    assert diagnostics.likelihood_distribution == "Gaussian"
    assert diagnostics.central_chi_squared is not None
    assert diagnostics.minimum_effective_to_reported_error_ratio >= 2.0
    assert diagnostics.maximum_effective_to_reported_error_ratio > 10.0


def test_student_t_predictive_diagnostics_use_scale_not_gaussian_chi_square():
    result = make_portable_noise_result(distribution="student_t")
    first = posterior_predictive_samples(result, random_seed=61)
    repeated = posterior_predictive_samples(result, random_seed=61)
    diagnostics = result.fit_diagnostics(random_seed=61)

    np.testing.assert_allclose(first, repeated)
    assert diagnostics.likelihood_distribution == "StudentT"
    assert "not a Gaussian" in diagnostics.likelihood_scale_description
    assert diagnostics.central_chi_squared is None
    assert diagnostics.reduced_chi_squared is None
