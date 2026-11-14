"""Joint Ly-alpha and C II inference with shared ISM velocity structure."""

from dataclasses import dataclass, fields

import numpy as np

from lyapy.inference.common import InferenceConfigurationBuilder, SpectrumData
from lyapy.inference.composable import prepare_observed_spectrum_data
from lyapy.inference.joint import JointForwardModel, JointSpectrumData
from lyapy.inference.workflow import InferenceProblem
from lyapy.jax_components import (
    AtomicTransitionAbsorber,
    CompositeAbsorptionModel,
    HydrogenDeuteriumAbsorber,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.observations import (
    StisTargetSpecification,
    load_stis_target_spectrum,
)
from lyapy.spectral_lines import (
    gaussian_emission_model_for_line_group,
    voigt_emission_model_for_line_group,
)


LYA_SPECTRUM_NAME = "Ly alpha"
CII_SPECTRUM_NAME = "C II"


@dataclass(frozen=True)
class LyaCiiModelDefaults:
    """Initial values and explicit uniform bounds for a Ly-alpha/C II fit."""

    stellar_velocity: float
    stellar_velocity_bounds: tuple[float, float]
    cii_stellar_velocity: float
    cii_stellar_velocity_bounds: tuple[float, float]
    ism_velocity: float = 0.0
    ism_velocity_bounds: tuple[float, float] = (-40.0, 40.0)
    lya_log_total_flux: float = -11.75
    lya_log_total_flux_bounds: tuple[float, float] = (-14.0, -10.0)
    lya_lorentz_fwhm: float = 8.0
    lya_lorentz_fwhm_bounds: tuple[float, float] = (1.0, 1000.0)
    lya_gaussian_fwhm: float = 90.0
    lya_gaussian_fwhm_bounds: tuple[float, float] = (1.0, 1000.0)
    h1_log_column: float = 18.0
    h1_log_column_bounds: tuple[float, float] = (17.0, 19.5)
    h1_doppler_b: float = 12.0
    h1_doppler_b_bounds: tuple[float, float] = (5.0, 30.0)
    cii_log_total_flux: float = -13.0
    cii_log_total_flux_bounds: tuple[float, float] = (-16.0, -11.0)
    cii_fwhm: float = 50.0
    cii_fwhm_bounds: tuple[float, float] = (5.0, 300.0)
    cii_1335a_log_weight: float = 0.0
    cii_1335a_log_weight_bounds: tuple[float, float] = (-2.0, 2.0)
    cii_1335b_log_weight: float = 0.0
    cii_1335b_log_weight_bounds: tuple[float, float] = (-2.0, 2.0)
    cii_log_column: float = 14.0
    cii_log_column_bounds: tuple[float, float] = (11.0, 18.0)
    cii_star_log_column: float = 13.0
    cii_star_log_column_bounds: tuple[float, float] = (10.0, 18.0)
    cii_doppler_b: float = 5.0
    cii_doppler_b_bounds: tuple[float, float] = (1.0, 25.0)

    def __post_init__(self):
        for item in fields(self):
            if item.name.endswith("_bounds"):
                continue
            value = float(getattr(self, item.name))
            bounds = getattr(self, f"{item.name}_bounds")
            lower, upper = (float(bound) for bound in bounds)
            if not np.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower >= upper
            ):
                raise ValueError(f"{item.name}_bounds must be finite and increasing")
            if not lower <= value <= upper:
                raise ValueError(f"{item.name} must lie within its bounds")


@dataclass(frozen=True)
class LyaCiiModelBundle:
    """Matched configuration and line-specific composable forward models."""

    configuration: object
    lya_model: ComposableBinnedModel
    cii_model: ComposableBinnedModel

    def joint_forward_model(self):
        return JointForwardModel.from_mapping(
            {
                LYA_SPECTRUM_NAME: self.lya_model,
                CII_SPECTRUM_NAME: self.cii_model,
            }
        )

    def problem(self, lya_data, cii_data, *, name="Ly-alpha and C II joint fit"):
        if not isinstance(lya_data, SpectrumData) or not isinstance(
            cii_data,
            SpectrumData,
        ):
            raise TypeError("lya_data and cii_data must be SpectrumData")
        data = JointSpectrumData.from_mapping(
            {
                LYA_SPECTRUM_NAME: lya_data,
                CII_SPECTRUM_NAME: cii_data,
            }
        )
        return InferenceProblem(
            data,
            self.configuration,
            self.joint_forward_model(),
            name=name,
        )


def build_lya_cii_configuration(defaults):
    """Build the standard parameter vector without hidden parameter sharing."""
    if not isinstance(defaults, LyaCiiModelDefaults):
        raise TypeError("defaults must be LyaCiiModelDefaults")
    builder = InferenceConfigurationBuilder()
    parameter_names = (
        "stellar_velocity",
        "cii_stellar_velocity",
        "ism_velocity",
        "lya_log_total_flux",
        "lya_lorentz_fwhm",
        "lya_gaussian_fwhm",
        "h1_log_column",
        "h1_doppler_b",
        "cii_log_total_flux",
        "cii_fwhm",
        "cii_1335a_log_weight",
        "cii_1335b_log_weight",
        "cii_log_column",
        "cii_star_log_column",
        "cii_doppler_b",
    )
    for name in parameter_names:
        builder.add_uniform(
            name,
            getattr(defaults, name),
            *getattr(defaults, f"{name}_bounds"),
        )
    return builder.build()


def build_lya_cii_model_bundle(
    lya_operator,
    cii_operator,
    defaults,
):
    """Build Ly-alpha and C II models sharing only the ISM velocity.

    H I and C II have independent column densities and Doppler parameters.
    Ground-state C II 1334 uses ``cii_log_column``. The two C II* 1335
    absorbers share ``cii_star_log_column`` because they arise from the same
    excited lower level. All three C II stellar emission strengths remain
    free; no optically thin emission ratio is silently imposed.
    """
    configuration = build_lya_cii_configuration(defaults)
    index = configuration.parameter_names.index
    lya_emission = voigt_emission_model_for_line_group(
        "lya",
        velocity_index=index("stellar_velocity"),
        gaussian_fwhm_index=index("lya_gaussian_fwhm"),
        lorentz_fwhm_index=index("lya_lorentz_fwhm"),
        log_total_flux_index=index("lya_log_total_flux"),
    )
    lya_absorption = CompositeAbsorptionModel(
        (
            HydrogenDeuteriumAbsorber(
                index("h1_log_column"),
                index("h1_doppler_b"),
                index("ism_velocity"),
            ),
        )
    )
    cii_emission = gaussian_emission_model_for_line_group(
        "c_ii_1334_1335",
        velocity_index=index("cii_stellar_velocity"),
        fwhm_index=index("cii_fwhm"),
        log_total_flux_index=index("cii_log_total_flux"),
        log_relative_weight_indices=(
            index("cii_1335a_log_weight"),
            index("cii_1335b_log_weight"),
        ),
    )
    cii_absorption = CompositeAbsorptionModel(
        (
            AtomicTransitionAbsorber(
                "c_ii_1334",
                index("cii_log_column"),
                index("cii_doppler_b"),
                index("ism_velocity"),
            ),
            AtomicTransitionAbsorber(
                "c_ii_1335a",
                index("cii_star_log_column"),
                index("cii_doppler_b"),
                index("ism_velocity"),
            ),
            AtomicTransitionAbsorber(
                "c_ii_1335b",
                index("cii_star_log_column"),
                index("cii_doppler_b"),
                index("ism_velocity"),
            ),
        )
    )
    return LyaCiiModelBundle(
        configuration,
        ComposableBinnedModel(
            lya_emission,
            lya_operator,
            absorption_model=lya_absorption,
        ),
        ComposableBinnedModel(
            cii_emission,
            cii_operator,
            absorption_model=cii_absorption,
        ),
    )


def prepare_e140m_lya_cii_data(
    target,
    *,
    lya_window=None,
    cii_window=(1333.75, 1336.45),
    mask_target_airglow=True,
):
    """Load raw native-grid Ly-alpha and C II windows for one E140M target."""
    if not isinstance(target, StisTargetSpecification):
        raise TypeError("target must be a StisTargetSpecification")
    if not str(target.grating).upper().startswith("E140M"):
        raise ValueError("target must use STIS E140M")
    spectrum = load_stis_target_spectrum(target)
    lya_window = target.fit_window if lya_window is None else lya_window
    lya_exclusions = target.excluded_windows if mask_target_airglow else ()
    lya_data = prepare_observed_spectrum_data(
        spectrum,
        lya_window,
        exclusion_windows=lya_exclusions,
        uncertainty_scale=target.uncertainty_scale,
    )
    cii_data = prepare_observed_spectrum_data(
        spectrum,
        cii_window,
        uncertainty_scale=target.uncertainty_scale,
    )
    return lya_data, cii_data
