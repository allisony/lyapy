from dataclasses import dataclass
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
from lyapy.inference.joint import (
    JointForwardModel,
    JointLikelihood,
    JointSpectrumData,
    NamedForwardModel,
    NamedLikelihood,
    NamedSpectrumData,
    evaluate_joint_numpy_log_likelihoods,
)
from lyapy.inference.noise import GaussianNoiseModel, StudentTNoiseModel
from lyapy.inference.numpyro_backend import numpyro_spectrum_model, run_nuts
from lyapy.likelihoods import gaussian_log_likelihood


class Convolved(NamedTuple):
    observed: object


class Result(NamedTuple):
    convolved: Convolved


@dataclass(frozen=True)
class ConstantLineModel:
    shared_index: int
    offset_index: int

    def __call__(self, wavelength, parameters, convolution_kernel):
        del convolution_kernel
        level = parameters[self.shared_index] + parameters[self.offset_index]
        return Result(Convolved(jnp.full_like(wavelength, level)))


def spectrum(wavelength, flux, error):
    return SpectrumData(
        np.asarray(wavelength),
        np.asarray(flux),
        np.asarray(error),
        np.array([1.0]),
    )


def make_joint_problem():
    first = spectrum(
        [1.0, 2.0, 3.0],
        [1.1, 1.08, 1.12],
        [0.05, 0.05, 0.05],
    )
    second = spectrum(
        [10.0, 11.0, 12.0, 13.0],
        [0.8, 0.82, 0.78, 0.81],
        [0.04, 0.04, 0.04, 0.04],
    )
    data = JointSpectrumData(
        (
            NamedSpectrumData("Ly alpha", first),
            NamedSpectrumData("Si III", second),
        )
    )
    forward_model = JointForwardModel(
        (
            NamedForwardModel("Ly alpha", ConstantLineModel(0, 1)),
            NamedForwardModel("Si III", ConstantLineModel(0, 2)),
        )
    )
    configuration = InferenceConfiguration(
        base_parameters=(1.0, 0.1, -0.2),
        free_parameters=(
            UniformParameter("shared_level", 0, 0.5, 1.5),
            UniformParameter("lya_offset", 1, -0.4, 0.4),
            UniformParameter("si3_offset", 2, -0.4, 0.4),
        ),
        parameter_names=("shared_level", "lya_offset", "si3_offset"),
    )
    return data, configuration, forward_model


def test_joint_forward_model_handles_different_spectrum_lengths():
    data, configuration, forward_model = make_joint_problem()
    predictions = forward_model.predict(
        data,
        jnp.asarray(configuration.base_parameters),
    )

    assert tuple(prediction.shape for prediction in predictions) == ((3,), (4,))
    np.testing.assert_allclose(predictions[0], 1.1)
    np.testing.assert_allclose(predictions[1], 0.8)
    assert tuple(spectrum.site_name for spectrum in data.spectra) == (
        "Ly_alpha",
        "Si_III",
    )


def test_joint_dynesty_log_likelihood_is_sum_of_spectrum_likelihoods():
    data, configuration, forward_model = make_joint_problem()
    log_likelihood, _ = build_dynesty_functions(
        data,
        configuration,
        forward_model=forward_model,
    )
    free_values = np.asarray(configuration.base_parameters)
    predictions = forward_model.predict(data, free_values)
    expected = sum(
        gaussian_log_likelihood(
            named.data.flux,
            prediction,
            named.data.error,
        )
        for named, prediction in zip(data.spectra, predictions)
    )

    np.testing.assert_allclose(log_likelihood(free_values), expected, rtol=1e-13)


def test_joint_likelihood_allows_different_noise_models_by_spectrum():
    data, configuration, forward_model = make_joint_problem()
    likelihood = JointLikelihood(
        (
            NamedLikelihood("Ly alpha", GaussianNoiseModel()),
            NamedLikelihood(
                "Si III",
                StudentTNoiseModel(degrees_of_freedom=4.0),
            ),
        )
    )
    parameters = np.asarray(configuration.base_parameters)
    predictions = forward_model.predict(data, parameters)
    contributions = evaluate_joint_numpy_log_likelihoods(
        likelihood,
        data,
        predictions,
        parameters,
    )

    assert len(contributions) == 2
    assert np.all(np.isfinite(contributions))
    report = ModelHypothesis(
        "joint",
        configuration,
        forward_model=forward_model,
        likelihood_function=likelihood,
    ).as_dict()
    assert report["forward_model_specification"]["type"] == "joint"
    assert (
        report["likelihood_specification"]["spectra"]["Si III"]
        ["settings"]["distribution"]
        == "StudentT"
    )


def test_joint_dynesty_and_numpyro_log_densities_agree():
    data, configuration, forward_model = make_joint_problem()
    free_values = np.asarray(configuration.base_parameters)
    dynesty_log_likelihood, _ = build_dynesty_functions(
        data,
        configuration,
        forward_model=forward_model,
    )
    values = {
        parameter.name: free_values[position]
        for position, parameter in enumerate(configuration.free_parameters)
    }
    model = partial(
        numpyro_spectrum_model,
        forward_model=forward_model,
    )
    numpyro_log_density, _ = log_density(
        model,
        (data, configuration),
        {},
        values,
    )
    uniform_log_prior = -sum(
        np.log(parameter.upper - parameter.lower)
        for parameter in configuration.free_parameters
    )

    np.testing.assert_allclose(
        numpyro_log_density,
        dynesty_log_likelihood(free_values) + uniform_log_prior,
        rtol=1e-13,
    )


def test_numpyro_joint_fit_records_each_detector_prediction():
    data, configuration, forward_model = make_joint_problem()
    sampler = run_nuts(
        data,
        configuration,
        jax.random.key(47),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        forward_model=forward_model,
    )
    samples = sampler.get_samples()

    assert samples["shared_level"].shape == (10,)
    assert samples["model_flux_Ly_alpha"].shape == (10, 3)
    assert samples["model_flux_Si_III"].shape == (10, 4)


def test_joint_names_must_match_and_have_unique_sample_sites():
    data, configuration, forward_model = make_joint_problem()
    mismatched = JointForwardModel(
        (
            NamedForwardModel("Ly alpha", ConstantLineModel(0, 1)),
            NamedForwardModel("N V", ConstantLineModel(0, 2)),
        )
    )
    with pytest.raises(ValueError, match="names must match"):
        mismatched.predict(data, jnp.asarray(configuration.base_parameters))
    with pytest.raises(ValueError, match="unique site names"):
        JointSpectrumData(
            (
                NamedSpectrumData("Si III", data.spectra[0].data),
                NamedSpectrumData("Si-III", data.spectra[1].data),
            )
        )
