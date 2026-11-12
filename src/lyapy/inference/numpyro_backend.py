"""NumPyro/NUTS adapter for the differentiable Lyman-alpha model."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from lyapy.jax_models import evaluate_voigt_lya_model_jax


def numpyro_spectrum_model(data, configuration):
    """NumPyro model using the shared uniform-prior configuration."""
    free_values = []
    for parameter in configuration.free_parameters:
        free_values.append(
            numpyro.sample(
                parameter.name,
                dist.Uniform(parameter.lower, parameter.upper),
            )
        )
    full_parameters = configuration.assemble_jax(jnp.stack(free_values))
    prediction = evaluate_voigt_lya_model_jax(
        jnp.asarray(data.wavelength),
        full_parameters,
        jnp.asarray(data.convolution_kernel),
    ).convolved.observed
    numpyro.deterministic("model_flux", prediction)
    numpyro.sample(
        "observed_flux",
        dist.Normal(prediction, jnp.asarray(data.error)),
        obs=jnp.asarray(data.flux),
    )


def run_nuts(
    data,
    configuration,
    rng_key,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    progress_bar=True,
):
    """Run NUTS initialized at the configuration's baseline values."""
    initial_values = {
        parameter.name: configuration.base_parameters[parameter.index]
        for parameter in configuration.free_parameters
    }
    kernel = NUTS(
        numpyro_spectrum_model,
        init_strategy=init_to_value(values=initial_values),
    )
    sampler = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    sampler.run(rng_key, data=data, configuration=configuration)
    return sampler
