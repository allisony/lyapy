"""Explicit nuisance parameters for a calibrated tabulated LSF."""

from dataclasses import replace
import math

from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.jax_instrument import parameterize_lsf_width


def augment_configuration_with_lsf_width(
    configuration,
    forward_model,
    *,
    width_scale_bounds=(0.8, 1.25),
    base_width_scale=1.0,
):
    """Append an LSF width scale and return a matched forward model.

    The fitted profile is ``LSF(offset / scale)`` and is renormalized on the
    high-resolution grid. A value of one reproduces the calibrated tabulation;
    values below or above one narrow or broaden its full shape, including its
    wings. The parameter is sampled in natural-log space.
    """
    if "ln_lsf_width_scale" in configuration.parameter_names:
        raise ValueError("configuration already contains ln_lsf_width_scale")
    lower, upper = width_scale_bounds
    if lower <= 0.0 or upper <= lower:
        raise ValueError("width_scale_bounds must be positive and increasing")
    if not lower <= base_width_scale <= upper:
        raise ValueError("base_width_scale must lie within width_scale_bounds")
    if not hasattr(forward_model, "observation_operator"):
        raise TypeError("forward_model must define observation_operator")

    index = len(configuration.parameter_names)
    parameter_name = "ln_lsf_width_scale"
    augmented = InferenceConfiguration(
        base_parameters=(
            *configuration.base_parameters,
            math.log(base_width_scale),
        ),
        free_parameters=(
            *configuration.free_parameters,
            UniformParameter(
                parameter_name,
                index,
                math.log(lower),
                math.log(upper),
            ),
        ),
        parameter_names=(*configuration.parameter_names, parameter_name),
    )
    parameterized_operator = parameterize_lsf_width(
        forward_model.observation_operator,
        index,
        width_scale_bounds=width_scale_bounds,
    )
    return augmented, replace(
        forward_model,
        observation_operator=parameterized_operator,
    )
