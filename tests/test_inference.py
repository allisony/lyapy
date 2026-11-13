import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = pytest.importorskip("jax.numpy")
numpyro = pytest.importorskip("numpyro")
pytest.importorskip("dynesty")

from numpyro.infer.util import log_density

from lyapy.inference.common import (
    GJ411_STELLAR_CONFIGURATION,
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
from lyapy.inference.reconstruction import (
    evaluate_reconstruction_samples,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.likelihoods import gaussian_log_likelihood
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


def test_configuration_supports_model_specific_parameter_schema():
    configuration = InferenceConfiguration(
        base_parameters=(0.0, 2.0, 5.0),
        free_parameters=(UniformParameter("width", 1, 0.5, 4.0),),
        parameter_names=("center", "width", "amplitude"),
    )

    np.testing.assert_allclose(configuration.assemble_numpy([3.0]), [0.0, 3.0, 5.0])
    assert configuration.parameter_names == ("center", "width", "amplitude")


def test_configuration_validates_model_specific_parameter_schema():
    with pytest.raises(ValueError, match="matching lengths"):
        InferenceConfiguration(
            base_parameters=(0.0,),
            free_parameters=(UniformParameter("center", 0, -1.0, 1.0),),
            parameter_names=("center", "width"),
        )
    with pytest.raises(ValueError, match="does not match"):
        InferenceConfiguration(
            base_parameters=(0.0,),
            free_parameters=(UniformParameter("width", 0, -1.0, 1.0),),
            parameter_names=("center",),
        )


def test_stellar_configuration_selects_identifiability_block():
    assert tuple(
        parameter.name
        for parameter in GJ411_STELLAR_CONFIGURATION.free_parameters
    ) == (
        "stellar_velocity",
        "log_amplitude",
        "lorentz_fwhm",
        "gaussian_fwhm",
        "reversal_depth",
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


def test_dynesty_adapter_accepts_hypothesis_specific_model_and_likelihood(
    synthetic_problem,
):
    data, configuration, truth = synthetic_problem

    def alternate_forward_model(wavelength, parameters, kernel):
        return evaluate_voigt_lya_model_jax(wavelength, parameters, kernel)

    def shifted_likelihood(observed, model, error):
        return gaussian_log_likelihood(observed, model, error) + 2.5

    baseline, _ = build_dynesty_functions(data, configuration)
    alternate, _ = build_dynesty_functions(
        data,
        configuration,
        forward_model=alternate_forward_model,
        likelihood_function=shifted_likelihood,
    )

    np.testing.assert_allclose(alternate(truth), baseline(truth) + 2.5)


def test_numpyro_nuts_smoke_run(synthetic_problem):
    data, configuration, _ = synthetic_problem
    sampler = run_nuts(
        data,
        configuration,
        jax.random.key(17),
        num_warmup=20,
        num_samples=20,
        progress_bar=False,
        dense_mass=True,
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
        sample="rwalk",
        bootstrap=0,
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


def test_posterior_reconstruction_profiles_and_integrated_flux(synthetic_problem):
    data, configuration, truth = synthetic_problem
    free_values = np.vstack((truth, truth + np.array([0.02, 0.05])))

    reconstruction = evaluate_reconstruction_samples(
        data,
        configuration,
        free_values,
    )

    assert reconstruction.intrinsic.shape == (2, data.wavelength.size)
    assert reconstruction.convolved_observed.shape == (2, data.wavelength.size)
    np.testing.assert_allclose(
        reconstruction.integrated_intrinsic_flux,
        np.trapezoid(reconstruction.intrinsic, data.wavelength, axis=1),
    )


def test_reconstruction_summaries_support_weighted_samples():
    profiles = np.array([[0.0, 1.0], [1.0, 2.0], [10.0, 20.0]])
    weights = np.array([0.05, 0.05, 0.9])

    unweighted = posterior_percentile_bands(profiles, probabilities=(0.5,))
    weighted = posterior_percentile_bands(
        profiles,
        probabilities=(0.5,),
        weights=weights,
    )
    flux_summary = summarize_integrated_flux(
        np.array([1.0, 2.0, 10.0]),
        probabilities=(0.5,),
        weights=weights,
    )

    np.testing.assert_allclose(unweighted, [[1.0, 2.0]])
    assert np.all(weighted > unweighted)
    assert flux_summary["percentiles"][0] > 2.0
