from functools import partial

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
numpyro = pytest.importorskip("numpyro")
from numpyro.infer.util import log_density

from lyapy.inference.common import make_reversal_comparison_configurations
from lyapy.inference.composable import (
    GJ411_COMPOSABLE_CONFIGURATION,
    load_gj411_composable_problem,
)
from lyapy.inference.dynesty_backend import build_dynesty_functions
from lyapy.inference.numpyro_backend import numpyro_spectrum_model, run_nuts
from lyapy.inference.reconstruction import (
    ReconstructionEvaluator,
    evaluate_reconstruction_samples,
)
from lyapy.inference.workflow import InferenceProblem


@pytest.fixture(scope="module")
def composable_problem():
    return load_gj411_composable_problem(oversample=4, model_padding=0.5)


def test_composable_problem_has_matched_data_configuration_and_model(
    composable_problem,
):
    problem = composable_problem
    parameters = jnp.asarray(problem.configuration.base_parameters)
    result = problem.forward_model(
        problem.data.wavelength,
        parameters,
        problem.data.convolution_kernel,
    )

    assert problem.configuration.parameter_names == (
        "stellar_velocity",
        "log_total_flux",
        "lorentz_fwhm",
        "gaussian_fwhm",
        "h1_column",
        "h1_doppler_b",
        "h1_velocity",
        "reversal_depth",
    )
    assert result.convolved.observed.shape == problem.data.wavelength.shape
    assert result.model_wavelength.size > problem.data.wavelength.size
    assert np.all(np.isfinite(result.convolved.observed))
    assert isinstance(problem, InferenceProblem)
    assert problem.name == "GJ 411 composable"


def test_generic_reversal_configurations_support_composable_schema():
    self_reversed, no_reversal = make_reversal_comparison_configurations(
        GJ411_COMPOSABLE_CONFIGURATION,
    )

    assert self_reversed.parameter_names[7] == "reversal_depth"
    assert no_reversal.base_parameters[7] == 0.0
    assert all(
        parameter.name != "reversal_depth"
        for parameter in no_reversal.free_parameters
    )


def test_dynesty_and_numpyro_use_same_composable_likelihood(composable_problem):
    problem = composable_problem
    configuration = problem.configuration
    free_values = np.asarray(
        [
            configuration.base_parameters[parameter.index]
            for parameter in configuration.free_parameters
        ]
    )
    dynesty_log_likelihood, _ = build_dynesty_functions(
        problem.data,
        configuration,
        forward_model=problem.forward_model,
    )
    numpyro_values = {
        parameter.name: free_values[position]
        for position, parameter in enumerate(configuration.free_parameters)
    }
    model = partial(
        numpyro_spectrum_model,
        forward_model=problem.forward_model,
    )
    numpyro_log_density, _ = log_density(
        model,
        (problem.data, configuration),
        {},
        numpyro_values,
    )
    uniform_log_prior = -sum(
        np.log(parameter.upper - parameter.lower)
        for parameter in configuration.free_parameters
    )

    np.testing.assert_allclose(
        numpyro_log_density,
        dynesty_log_likelihood(free_values) + uniform_log_prior,
        rtol=2e-12,
    )


def test_composable_reconstruction_uses_direct_total_flux(composable_problem):
    problem = composable_problem
    configuration = problem.configuration
    truth = np.asarray(
        [
            configuration.base_parameters[parameter.index]
            for parameter in configuration.free_parameters
        ]
    )
    values = np.vstack((truth, truth + np.array([0, 0.1, 0, 0, 0, 0, 0, 0])))
    reconstruction = evaluate_reconstruction_samples(
        problem.data,
        configuration,
        values,
        forward_model=problem.forward_model,
    )

    assert reconstruction.intrinsic.shape[1] == reconstruction.model_wavelength.size
    assert reconstruction.convolved_observed.shape == (2, problem.data.wavelength.size)
    np.testing.assert_allclose(
        reconstruction.integrated_intrinsic_flux,
        10.0 ** values[:, 1],
        rtol=2e-13,
    )

    evaluator = ReconstructionEvaluator(
        problem.data,
        configuration,
        forward_model=problem.forward_model,
        batch_size=1,
    )
    chunked = evaluator.evaluate(values)
    repeated = evaluator.evaluate(values)
    np.testing.assert_allclose(chunked.intrinsic, reconstruction.intrinsic)
    np.testing.assert_allclose(
        chunked.convolved_observed,
        reconstruction.convolved_observed,
    )
    np.testing.assert_allclose(
        chunked.integrated_intrinsic_flux,
        reconstruction.integrated_intrinsic_flux,
    )
    np.testing.assert_allclose(repeated.intrinsic, chunked.intrinsic)

    with pytest.raises(ValueError, match="batch_size"):
        evaluate_reconstruction_samples(
            problem.data,
            configuration,
            values,
            forward_model=problem.forward_model,
            batch_size=0,
        )


def test_numpyro_accepts_composable_forward_model(composable_problem):
    problem = composable_problem
    reduced_configuration = type(problem.configuration)(
        base_parameters=problem.configuration.base_parameters,
        free_parameters=(problem.configuration.free_parameters[1],),
        parameter_names=problem.configuration.parameter_names,
    )
    sampler = run_nuts(
        problem.data,
        reduced_configuration,
        jax.random.key(29),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
        forward_model=problem.forward_model,
    )
    samples = sampler.get_samples()

    assert samples["log_total_flux"].shape == (10,)
    assert samples["model_flux"].shape == (10, problem.data.wavelength.size)
    assert np.all(np.isfinite(samples["model_flux"]))
