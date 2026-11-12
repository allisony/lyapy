"""dynesty adapter using the same JAX model and uniform priors as NumPyro."""

import jax
import jax.numpy as jnp
import numpy as np

from lyapy.jax_models import evaluate_voigt_lya_model_jax


def build_dynesty_functions(data, configuration):
    """Return dynesty-compatible likelihood and prior-transform callables."""
    wave = jnp.asarray(data.wavelength)
    observed = np.asarray(data.flux)
    error = np.asarray(data.error)
    kernel = jnp.asarray(data.convolution_kernel)

    @jax.jit
    def predict(free_values):
        full_parameters = configuration.assemble_jax(free_values)
        return evaluate_voigt_lya_model_jax(
            wave, full_parameters, kernel
        ).convolved.observed

    def log_likelihood(free_values):
        model_flux = np.asarray(predict(jnp.asarray(free_values)))
        normalized_residual = (observed - model_flux) / error
        return float(
            -0.5
            * np.sum(normalized_residual**2 + np.log(2.0 * np.pi * error**2))
        )

    return log_likelihood, configuration.transform_unit_cube


def run_dynesty(
    data,
    configuration,
    nlive=500,
    dlogz=0.01,
    random_state=None,
    print_progress=True,
):
    """Run static nested sampling and return the dynesty sampler."""
    from dynesty import NestedSampler

    log_likelihood, prior_transform = build_dynesty_functions(data, configuration)
    sampler = NestedSampler(
        log_likelihood,
        prior_transform,
        configuration.ndim,
        nlive=nlive,
        rstate=random_state,
    )
    sampler.run_nested(dlogz=dlogz, print_progress=print_progress)
    return sampler
