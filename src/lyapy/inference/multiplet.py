"""Matched hypotheses for testing fixed versus free multiplet ratios."""

from dataclasses import dataclass
import math

from lyapy.inference.common import InferenceConfiguration
from lyapy.inference.comparison import ModelHypothesis
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.likelihoods import gaussian_log_likelihood
from lyapy.spectral_lines import (
    gaussian_emission_model_for_line_group,
    get_spectral_line_group,
    voigt_emission_model_for_line_group,
)


@dataclass(frozen=True)
class MultipletForwardModel:
    """Self-describing multiplet model compatible with inference backends."""

    model: ComposableBinnedModel
    line_group: object
    profile: str
    ratio_mode: str
    fixed_relative_weights: tuple[float, ...] | None = None
    log_relative_weight_parameter_names: tuple[str, ...] = ()

    def __call__(self, wavelength, parameters, convolution_kernel):
        return self.model(wavelength, parameters, convolution_kernel)

    def as_dict(self):
        ratio = {"mode": self.ratio_mode}
        if self.fixed_relative_weights is not None:
            ratio["relative_weights"] = list(self.fixed_relative_weights)
        if self.log_relative_weight_parameter_names:
            ratio["log10_parameter_names"] = list(
                self.log_relative_weight_parameter_names
            )
        return {
            "type": "multiplet",
            "profile": self.profile,
            "line_group": self.line_group.as_dict(),
            "component_ratio": ratio,
            "includes_absorption": self.model.absorption_model is not None,
        }


@dataclass(frozen=True)
class MultipletRatioHypotheses:
    """Matched fixed-ratio and free-ratio model hypotheses."""

    fixed_ratio: ModelHypothesis
    free_ratio: ModelHypothesis

    @property
    def hypotheses(self):
        return (self.fixed_ratio, self.free_ratio)

    def __iter__(self):
        return iter(self.hypotheses)


def _fixed_ratio_configuration(
    configuration,
    ratio_indices,
    fixed_relative_weights,
):
    ratio_indices = tuple(ratio_indices)
    ratio_index_set = set(ratio_indices)
    free_by_index = {
        parameter.index: parameter
        for parameter in configuration.free_parameters
    }
    missing = [index for index in ratio_indices if index not in free_by_index]
    if missing:
        raise ValueError(
            "free-ratio indices must identify free parameters in the supplied "
            f"configuration; missing indices: {missing}"
        )
    common_free_parameters = tuple(
        parameter
        for parameter in configuration.free_parameters
        if parameter.index not in ratio_index_set
    )
    if not common_free_parameters:
        raise ValueError(
            "the fixed-ratio hypothesis requires at least one common free parameter"
        )
    base_parameters = list(configuration.base_parameters)
    reference_weight = fixed_relative_weights[0]
    for index, weight in zip(ratio_indices, fixed_relative_weights[1:]):
        base_parameters[index] = math.log10(weight / reference_weight)
    return InferenceConfiguration(
        base_parameters=tuple(base_parameters),
        free_parameters=common_free_parameters,
        parameter_names=configuration.parameter_names,
    )


def make_multiplet_ratio_hypotheses(
    group,
    observation_operator,
    configuration,
    *,
    velocity_index,
    log_total_flux_index,
    gaussian_fwhm_index,
    log_relative_weight_indices,
    fixed_relative_weights,
    lorentz_fwhm_index=None,
    absorption_model=None,
    likelihood_function=gaussian_log_likelihood,
    fixed_name=None,
    free_name=None,
):
    """Build hypotheses differing only in fixed versus sampled line ratios.

    ``configuration`` must include one free log10 relative-weight parameter for
    every transition after the first. The fixed hypothesis removes only those
    parameters; all common priors and all forward-model operations are reused.
    Passing ``lorentz_fwhm_index`` selects Voigt components, otherwise Gaussian
    components are used. Prior choices remain entirely caller-supplied.
    """
    if not isinstance(configuration, InferenceConfiguration):
        raise TypeError("configuration must be an InferenceConfiguration")
    group = get_spectral_line_group(group)
    number = len(group.transitions)
    if number < 2:
        raise ValueError("multiplet-ratio comparison requires multiple transitions")
    ratio_indices = tuple(log_relative_weight_indices)
    fixed_weights = tuple(float(weight) for weight in fixed_relative_weights)
    if len(ratio_indices) != number - 1:
        raise ValueError(
            "one log relative-weight index is required after the first line"
        )
    if len(fixed_weights) != number or any(
        not math.isfinite(weight) or weight <= 0.0
        for weight in fixed_weights
    ):
        raise ValueError(
            "fixed_relative_weights must contain one positive value per transition"
        )
    if len(ratio_indices) != len(set(ratio_indices)):
        raise ValueError("log relative-weight indices must be unique")
    if any(
        index < 0 or index >= len(configuration.parameter_names)
        for index in ratio_indices
    ):
        raise ValueError(
            "log relative-weight indices must lie within the parameter vector"
        )
    profile_indices = {
        velocity_index,
        log_total_flux_index,
        gaussian_fwhm_index,
    }
    if lorentz_fwhm_index is not None:
        profile_indices.add(lorentz_fwhm_index)
    if any(
        index < 0 or index >= len(configuration.parameter_names)
        for index in profile_indices
    ):
        raise ValueError("profile indices must lie within the parameter vector")
    if set(ratio_indices) & profile_indices:
        raise ValueError(
            "relative-weight parameters must be distinct from profile parameters"
        )

    fixed_configuration = _fixed_ratio_configuration(
        configuration,
        ratio_indices,
        fixed_weights,
    )
    ratio_parameter_names = tuple(
        configuration.parameter_names[index]
        for index in ratio_indices
    )
    if lorentz_fwhm_index is None:
        profile = "gaussian"
        fixed_emission = gaussian_emission_model_for_line_group(
            group,
            velocity_index,
            gaussian_fwhm_index,
            log_total_flux_index,
            relative_weights=fixed_weights,
        )
        free_emission = gaussian_emission_model_for_line_group(
            group,
            velocity_index,
            gaussian_fwhm_index,
            log_total_flux_index,
            log_relative_weight_indices=ratio_indices,
        )
    else:
        profile = "voigt"
        fixed_emission = voigt_emission_model_for_line_group(
            group,
            velocity_index,
            gaussian_fwhm_index,
            lorentz_fwhm_index,
            log_total_flux_index,
            relative_weights=fixed_weights,
        )
        free_emission = voigt_emission_model_for_line_group(
            group,
            velocity_index,
            gaussian_fwhm_index,
            lorentz_fwhm_index,
            log_total_flux_index,
            log_relative_weight_indices=ratio_indices,
        )

    fixed_model = MultipletForwardModel(
        ComposableBinnedModel(
            fixed_emission,
            observation_operator,
            absorption_model=absorption_model,
        ),
        group,
        profile,
        "fixed",
        fixed_relative_weights=fixed_weights,
    )
    free_model = MultipletForwardModel(
        ComposableBinnedModel(
            free_emission,
            observation_operator,
            absorption_model=absorption_model,
        ),
        group,
        profile,
        "free",
        log_relative_weight_parameter_names=ratio_parameter_names,
    )
    prefix = group.key
    return MultipletRatioHypotheses(
        fixed_ratio=ModelHypothesis(
            fixed_name or f"{prefix}_fixed_ratio",
            fixed_configuration,
            forward_model=fixed_model,
            likelihood_function=likelihood_function,
        ),
        free_ratio=ModelHypothesis(
            free_name or f"{prefix}_free_ratio",
            configuration,
            forward_model=free_model,
            likelihood_function=likelihood_function,
        ),
    )
