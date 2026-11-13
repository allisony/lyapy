"""NumPyro/NUTS adapter for differentiable Lyman-alpha models."""

from functools import partial

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.inference.common import TruncatedNormalParameter, UniformParameter
from lyapy.inference.noise import evaluate_jax_log_likelihood
from lyapy.likelihoods import gaussian_log_likelihood


def _prior_distribution(parameter):
    if isinstance(parameter, UniformParameter):
        return dist.Uniform(parameter.lower, parameter.upper)
    if isinstance(parameter, TruncatedNormalParameter):
        return dist.TruncatedNormal(
            parameter.mean,
            parameter.standard_deviation,
            low=parameter.lower,
            high=parameter.upper,
        )
    raise TypeError(f"unsupported prior type: {type(parameter).__name__}")


def numpyro_spectrum_model(
    data,
    configuration,
    forward_model=evaluate_voigt_lya_model_jax,
    likelihood_function=gaussian_log_likelihood,
):
    """NumPyro model using a shared configuration and forward model."""
    free_values = []
    for parameter in configuration.free_parameters:
        free_values.append(
            numpyro.sample(
                parameter.name,
                _prior_distribution(parameter),
            )
        )
    full_parameters = configuration.assemble_jax(jnp.stack(free_values))
    prediction = forward_model(
        jnp.asarray(data.wavelength),
        full_parameters,
        jnp.asarray(data.convolution_kernel),
    ).convolved.observed
    numpyro.deterministic("model_flux", prediction)
    numpyro.factor(
        "observed_flux_log_likelihood",
        evaluate_jax_log_likelihood(
            likelihood_function,
            jnp.asarray(data.flux),
            prediction,
            jnp.asarray(data.error),
            full_parameters,
        ),
    )


def run_nuts(
    data,
    configuration,
    rng_key,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
    target_accept_probability=0.8,
    chain_method="parallel",
    dense_mass=False,
    forward_model=evaluate_voigt_lya_model_jax,
    likelihood_function=gaussian_log_likelihood,
):
    """Run NUTS initialized at the configuration's baseline values.

    The returned sampler retains divergence, acceptance-probability, and tree-size
    diagnostics for use by :func:`lyapy.inference.diagnostics.summarize_nuts`.
    """
    initial_values = {
        parameter.name: configuration.base_parameters[parameter.index]
        for parameter in configuration.free_parameters
    }
    model = partial(
        numpyro_spectrum_model,
        forward_model=forward_model,
        likelihood_function=likelihood_function,
    )
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=initial_values),
        target_accept_prob=target_accept_probability,
        dense_mass=dense_mass,
    )
    sampler = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
        chain_method=chain_method,
    )
    sampler.run(
        rng_key,
        data=data,
        configuration=configuration,
        extra_fields=("diverging", "accept_prob", "num_steps"),
    )
    return sampler
