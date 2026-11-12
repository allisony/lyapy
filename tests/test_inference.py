import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
pytest.importorskip("dynesty")

from numpyro.infer.util import log_density

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    TruncatedNormalParameter,
    UniformParameter,
)
from lyapy.inference.dynesty_backend import build_dynesty_functions, run_dynesty
from lyapy.inference.diagnostics import (
    summarize_dynesty,
    summarize_nuts,
    weighted_quantile,
)
from lyapy.inference.numpyro_backend import numpyro_spectrum_model, run_nuts
from lyapy.jax_models import evaluate_voigt_lya_model_jax
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


@pytest.fixture(scope="module")
def synthetic_problem():
    reference = evaluate_gj411_legacy_case()
    wave = np.asarray(reference["wave"], dtype=np.float64)
    kernel = np.asarray(reference["lsf_kernel"], dtype=np.float64)
    parameters = np.asarray(PARAMETERS, dtype=np.float64)
    model_flux = np.asarray(
        evaluate_voigt_lya_model_jax(wave, parameters, kernel).convolved.observed
    )
    error = np.full_like(model_flux, 5e-14)
    data = SpectrumData(wave, model_flux, error, kernel)
    configuration = InferenceConfiguration(
        base_parameters=tuple(parameters),
        free_parameters=(
            UniformParameter("log_amplitude", 1, -10.3, -9.8),
            UniformParameter("reversal_depth", 11, 1.0, 1.9),
        ),
    )
    truth = np.array([parameters[1], parameters[11]])
    return data, configuration, truth


def test_shared_configuration_assembles_and_transforms_parameters(
    synthetic_problem,
):
    _, configuration, truth = synthetic_problem
    midpoint = configuration.transform_unit_cube(np.array([0.5, 0.5]))
    full_parameters = configuration.assemble_numpy(truth)

    np.testing.assert_allclose(midpoint, [-10.05, 1.45])
    np.testing.assert_allclose(full_parameters[[1, 11]], truth)
    np.testing.assert_allclose(
        configuration.assemble_jax(truth),
        full_parameters,
        rtol=0.0,
        atol=0.0,
    )


def test_numpyro_and_dynesty_log_densities_agree(synthetic_problem):
    data, configuration, truth = synthetic_problem
    log_likelihood, _ = build_dynesty_functions(data, configuration)
    numpyro_values = {
        parameter.name: truth[position]
        for position, parameter in enumerate(configuration.free_parameters)
    }
    numpyro_log_density, _ = log_density(
        numpyro_spectrum_model,
        (data, configuration),
        {},
        numpyro_values,
    )
    uniform_log_prior = -sum(
        np.log(parameter.upper - parameter.lower)
        for parameter in configuration.free_parameters
    )

    np.testing.assert_allclose(
        numpyro_log_density,
        log_likelihood(truth) + uniform_log_prior,
        rtol=1e-12,
    )


def test_truncated_normal_prior_is_shared_by_both_backends(synthetic_problem):
    from scipy.stats import truncnorm

    data, original_configuration, truth = synthetic_problem
    parameter = TruncatedNormalParameter(
        "log_amplitude",
        1,
        -10.3,
        -9.8,
        mean=-10.05,
        standard_deviation=0.1,
    )
    configuration = InferenceConfiguration(
        base_parameters=original_configuration.base_parameters,
        free_parameters=(parameter,),
    )
    value = truth[:1]
    log_likelihood, prior_transform = build_dynesty_functions(data, configuration)
    numpyro_log_density, _ = log_density(
        numpyro_spectrum_model,
        (data, configuration),
        {},
        {"log_amplitude": value[0]},
    )
    lower = (parameter.lower - parameter.mean) / parameter.standard_deviation
    upper = (parameter.upper - parameter.mean) / parameter.standard_deviation
    prior_log_density = truncnorm.logpdf(
        value[0],
        lower,
        upper,
        loc=parameter.mean,
        scale=parameter.standard_deviation,
    )

    np.testing.assert_allclose(prior_transform([0.5]), [parameter.mean])
    np.testing.assert_allclose(
        numpyro_log_density,
        log_likelihood(value) + prior_log_density,
        rtol=1e-12,
    )


def test_truth_has_higher_likelihood_than_offset_parameters(synthetic_problem):
    data, configuration, truth = synthetic_problem
    log_likelihood, prior_transform = build_dynesty_functions(data, configuration)

    assert log_likelihood(truth) > log_likelihood(truth + np.array([0.1, 0.2]))
    np.testing.assert_allclose(
        prior_transform(np.zeros(configuration.ndim)),
        [parameter.lower for parameter in configuration.free_parameters],
    )
    np.testing.assert_allclose(
        prior_transform(np.ones(configuration.ndim)),
        [parameter.upper for parameter in configuration.free_parameters],
    )


def test_numpyro_nuts_smoke_run(synthetic_problem):
    data, configuration, _ = synthetic_problem
    sampler = run_nuts(
        data,
        configuration,
        jax.random.key(17),
        num_warmup=20,
        num_samples=20,
        progress_bar=False,
    )
    samples = sampler.get_samples()

    assert {"log_amplitude", "reversal_depth"}.issubset(samples)
    assert samples["log_amplitude"].shape == (20,)
    assert samples["reversal_depth"].shape == (20,)
    assert samples["model_flux"].shape == (20, data.wavelength.size)
    assert all(np.all(np.isfinite(values)) for values in samples.values())

    diagnostics = summarize_nuts(
        sampler,
        ("log_amplitude", "reversal_depth"),
    )
    assert diagnostics["chains"] == 1
    assert diagnostics["draws_per_chain"] == 20
    assert diagnostics["divergences"] >= 0
    assert 0.0 <= diagnostics["mean_acceptance_probability"] <= 1.0


def test_dynesty_smoke_run(synthetic_problem):
    data, configuration, _ = synthetic_problem
    sampler = run_dynesty(
        data,
        configuration,
        nlive=20,
        dlogz=10.0,
        random_state=np.random.default_rng(23),
        print_progress=False,
    )

    assert sampler.results.samples.shape[1] == configuration.ndim
    assert np.all(np.isfinite(sampler.results.logl))

    diagnostics = summarize_dynesty(
        sampler,
        ("log_amplitude", "reversal_depth"),
    )
    assert diagnostics["posterior_effective_sample_size"] > 0.0
    assert diagnostics["likelihood_calls"] > 0
    assert np.isfinite(diagnostics["log_evidence"])


def test_weighted_quantile_validates_and_uses_weights():
    values = np.array([0.0, 1.0, 2.0])

    assert weighted_quantile(values, 0.5) == 1.0
    assert weighted_quantile(values, 0.5, [0.05, 0.05, 0.9]) > 1.0
    with pytest.raises(ValueError, match="positive sum"):
        weighted_quantile(values, 0.5, np.zeros(3))
