"""Small, extensible catalog of scientifically distinct stellar models."""

from dataclasses import dataclass

import numpy as np

from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.inference.comparison import ModelHypothesis
from lyapy.inference.composable import (
    COMPOSABLE_GJ411_BASE_PARAMETERS,
    build_e140m_observation_operator,
)
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    GaussianEmissionComponent,
    GaussianOpticalDepthReversal,
    HydrogenDeuteriumAbsorber,
    LegacyIntensityReversal,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.observations import StisTargetSpecification


LYA_REST_WAVELENGTH = 1215.67
GJ411_LOG_TOTAL_FLUX = COMPOSABLE_GJ411_BASE_PARAMETERS[1]


@dataclass(frozen=True)
class LyaCatalogDefaults:
    """Target-specific initial values and velocity prior bounds."""

    stellar_velocity: float
    log_total_flux: float
    stellar_velocity_bounds: tuple[float, float]
    reversal_velocity_bounds: tuple[float, float]
    narrow_velocity_bounds: tuple[float, float]
    broad_velocity_bounds: tuple[float, float]
    h1_column: float = 17.84
    h1_doppler_b: float = 11.84
    h1_velocity: float = 4.18
    h1_velocity_bounds: tuple[float, float] = (-30.0, 30.0)

    def __post_init__(self):
        scalar_values = (
            self.stellar_velocity,
            self.log_total_flux,
            self.h1_column,
            self.h1_doppler_b,
            self.h1_velocity,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("catalog initial values must be finite")
        for name in (
            "stellar_velocity_bounds",
            "reversal_velocity_bounds",
            "narrow_velocity_bounds",
            "broad_velocity_bounds",
            "h1_velocity_bounds",
        ):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ValueError(f"{name} must be increasing")
        if not (
            self.stellar_velocity_bounds[0]
            <= self.stellar_velocity
            <= self.stellar_velocity_bounds[1]
        ):
            raise ValueError("stellar_velocity must lie within its bounds")


GJ411_CATALOG_DEFAULTS = LyaCatalogDefaults(
    stellar_velocity=-87.0,
    log_total_flux=GJ411_LOG_TOTAL_FLUX,
    stellar_velocity_bounds=(-100.0, -75.0),
    reversal_velocity_bounds=(-120.0, -50.0),
    narrow_velocity_bounds=(-130.0, -40.0),
    broad_velocity_bounds=(-150.0, 0.0),
)


def catalog_defaults_for_target(target, *, initial_log_total_flux=-11.75):
    """Create broad, explicit starting bounds around a target's known RV."""
    if not isinstance(target, StisTargetSpecification):
        raise TypeError("target must be a StisTargetSpecification")
    velocity = float(target.stellar_radial_velocity)
    return LyaCatalogDefaults(
        stellar_velocity=velocity,
        log_total_flux=float(initial_log_total_flux),
        stellar_velocity_bounds=(velocity - 25.0, velocity + 25.0),
        reversal_velocity_bounds=(velocity - 40.0, velocity + 40.0),
        narrow_velocity_bounds=(velocity - 45.0, velocity + 45.0),
        broad_velocity_bounds=(velocity - 75.0, velocity + 75.0),
    )


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One named model, its priors, and its assembled forward model."""

    name: str
    description: str
    configuration: InferenceConfiguration
    forward_model: ComposableBinnedModel
    components: tuple[str, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name:
            raise ValueError("catalog model name must not be empty")
        if not self.description:
            raise ValueError("catalog model description must not be empty")
        if not self.components:
            raise ValueError("catalog model must describe at least one component")

    def hypothesis(self):
        """Return an evidence-framework hypothesis for this entry."""
        return ModelHypothesis(
            self.name,
            self.configuration,
            forward_model=self.forward_model,
        )

    def as_dict(self):
        """Return a JSON-compatible catalog and prior specification."""
        specification = self.hypothesis().as_dict()
        specification.update(
            description=self.description,
            components=list(self.components),
            tags=list(self.tags),
        )
        return specification


def make_model_catalog_entry(
    name,
    description,
    configuration,
    emission_model,
    observation_operator,
    *,
    absorption_model=None,
    components,
    tags=(),
):
    """Create a catalog entry from standard or user-defined components."""
    return ModelCatalogEntry(
        name=name,
        description=description,
        configuration=configuration,
        forward_model=ComposableBinnedModel(
            emission_model,
            observation_operator,
            absorption_model=absorption_model,
        ),
        components=tuple(components),
        tags=tuple(tags),
    )


def _configuration(parameter_names, base_parameters, priors):
    return InferenceConfiguration(
        base_parameters=tuple(base_parameters),
        free_parameters=tuple(
            UniformParameter(name, index, lower, upper)
            for index, (name, lower, upper) in enumerate(priors)
        ),
        parameter_names=tuple(parameter_names),
    )


def _single_gaussian(
    operator,
    *,
    gaussian_reversal=False,
    defaults=GJ411_CATALOG_DEFAULTS,
):
    if gaussian_reversal:
        names = (
            "stellar_velocity",
            "log_total_flux",
            "gaussian_fwhm",
            "reversal_velocity",
            "reversal_fwhm",
            "reversal_peak_optical_depth",
            "h1_column",
            "h1_doppler_b",
            "h1_velocity",
        )
        base = (
            defaults.stellar_velocity,
            defaults.log_total_flux,
            88.68,
            defaults.stellar_velocity,
            40.0,
            1.5,
            defaults.h1_column,
            defaults.h1_doppler_b,
            defaults.h1_velocity,
        )
        priors = (
            (names[0], *defaults.stellar_velocity_bounds),
            (names[1], -13.0, -10.5),
            (names[2], 1.0, 1000.0),
            (names[3], *defaults.reversal_velocity_bounds),
            (names[4], 1.0, 300.0),
            (names[5], 0.0, 5.0),
            (names[6], 17.5, 18.9),
            (names[7], 8.6, 20.0),
            (names[8], *defaults.h1_velocity_bounds),
        )
        reversal = GaussianOpticalDepthReversal(LYA_REST_WAVELENGTH, 3, 4, 5)
        absorber = HydrogenDeuteriumAbsorber(6, 7, 8)
        name = "gaussian_gaussian_reversal"
        description = "Single Gaussian emission with Gaussian optical-depth reversal"
        components = (
            "Gaussian emission",
            "Gaussian optical-depth reversal",
            "H I + D I",
        )
        tags = ("single-component", "gaussian", "self-reversed")
    else:
        names = (
            "stellar_velocity",
            "log_total_flux",
            "gaussian_fwhm",
            "h1_column",
            "h1_doppler_b",
            "h1_velocity",
        )
        base = (
            defaults.stellar_velocity,
            defaults.log_total_flux,
            88.68,
            defaults.h1_column,
            defaults.h1_doppler_b,
            defaults.h1_velocity,
        )
        priors = (
            (names[0], *defaults.stellar_velocity_bounds),
            (names[1], -13.0, -10.5),
            (names[2], 1.0, 1000.0),
            (names[3], 17.5, 18.9),
            (names[4], 8.6, 20.0),
            (names[5], *defaults.h1_velocity_bounds),
        )
        reversal = None
        absorber = HydrogenDeuteriumAbsorber(3, 4, 5)
        name = "single_gaussian"
        description = "Single Gaussian emission without self-reversal"
        components = ("Gaussian emission", "H I + D I")
        tags = ("single-component", "gaussian", "no-reversal")
    configuration = _configuration(names, base, priors)
    emission = CompositeEmissionModel(
        (GaussianEmissionComponent(LYA_REST_WAVELENGTH, 0, 2),),
        1,
        reversal=reversal,
    )
    absorption = CompositeAbsorptionModel((absorber,))
    return make_model_catalog_entry(
        name,
        description,
        configuration,
        emission,
        operator,
        absorption_model=absorption,
        components=components,
        tags=tags,
    )


def _single_voigt(
    operator,
    reversal_kind=None,
    *,
    defaults=GJ411_CATALOG_DEFAULTS,
):
    common_names = (
        "stellar_velocity",
        "log_total_flux",
        "lorentz_fwhm",
        "gaussian_fwhm",
    )
    common_base = (
        defaults.stellar_velocity,
        defaults.log_total_flux,
        7.76,
        88.68,
    )
    common_priors = (
        (common_names[0], *defaults.stellar_velocity_bounds),
        (common_names[1], -13.0, -10.5),
        (common_names[2], 1.0, 1000.0),
        (common_names[3], 1.0, 1000.0),
    )
    if reversal_kind is None:
        names = common_names + ("h1_column", "h1_doppler_b", "h1_velocity")
        base = common_base + (
            defaults.h1_column,
            defaults.h1_doppler_b,
            defaults.h1_velocity,
        )
        priors = common_priors + (
            (names[4], 17.5, 18.9),
            (names[5], 8.6, 20.0),
            (names[6], *defaults.h1_velocity_bounds),
        )
        reversal = None
        absorber = HydrogenDeuteriumAbsorber(4, 5, 6)
        name = "single_voigt"
        description = "Single Voigt emission without self-reversal"
        reversal_component = ()
        tags = ("single-component", "voigt", "no-reversal")
    elif reversal_kind == "legacy":
        names = common_names + (
            "h1_column",
            "h1_doppler_b",
            "h1_velocity",
            "reversal_depth",
        )
        base = common_base + (
            defaults.h1_column,
            defaults.h1_doppler_b,
            defaults.h1_velocity,
            1.5,
        )
        priors = common_priors + (
            (names[4], 17.5, 18.9),
            (names[5], 8.6, 20.0),
            (names[6], *defaults.h1_velocity_bounds),
            (names[7], 0.0, 1.8),
        )
        reversal = LegacyIntensityReversal(7)
        absorber = HydrogenDeuteriumAbsorber(4, 5, 6)
        name = "voigt_legacy_reversal"
        description = "Single Voigt emission with the legacy intensity-linked reversal"
        reversal_component = ("Legacy intensity-linked reversal",)
        tags = ("single-component", "voigt", "self-reversed", "legacy-reversal")
    elif reversal_kind == "gaussian":
        names = common_names + (
            "reversal_velocity",
            "reversal_fwhm",
            "reversal_peak_optical_depth",
            "h1_column",
            "h1_doppler_b",
            "h1_velocity",
        )
        base = common_base + (
            defaults.stellar_velocity,
            40.0,
            1.5,
            defaults.h1_column,
            defaults.h1_doppler_b,
            defaults.h1_velocity,
        )
        priors = common_priors + (
            (names[4], *defaults.reversal_velocity_bounds),
            (names[5], 1.0, 300.0),
            (names[6], 0.0, 5.0),
            (names[7], 17.5, 18.9),
            (names[8], 8.6, 20.0),
            (names[9], *defaults.h1_velocity_bounds),
        )
        reversal = GaussianOpticalDepthReversal(LYA_REST_WAVELENGTH, 4, 5, 6)
        absorber = HydrogenDeuteriumAbsorber(7, 8, 9)
        name = "voigt_gaussian_reversal"
        description = "Single Voigt emission with Gaussian optical-depth reversal"
        reversal_component = ("Gaussian optical-depth reversal",)
        tags = ("single-component", "voigt", "self-reversed", "gaussian-reversal")
    else:
        raise ValueError(f"unknown reversal_kind {reversal_kind!r}")
    configuration = _configuration(names, base, priors)
    emission = CompositeEmissionModel(
        (VoigtEmissionComponent(LYA_REST_WAVELENGTH, 0, 3, 2),),
        1,
        reversal=reversal,
    )
    absorption = CompositeAbsorptionModel((absorber,))
    return make_model_catalog_entry(
        name,
        description,
        configuration,
        emission,
        operator,
        absorption_model=absorption,
        components=("Voigt emission", *reversal_component, "H I + D I"),
        tags=tags,
    )


def _double_gaussian(operator, *, defaults=GJ411_CATALOG_DEFAULTS):
    names = (
        "narrow_velocity",
        "log_total_flux",
        "narrow_fwhm",
        "broad_velocity",
        "broad_fwhm",
        "log_broad_to_narrow_weight",
        "h1_column",
        "h1_doppler_b",
        "h1_velocity",
    )
    base = (
        defaults.stellar_velocity,
        defaults.log_total_flux,
        70.0,
        defaults.stellar_velocity,
        180.0,
        -0.3,
        defaults.h1_column,
        defaults.h1_doppler_b,
        defaults.h1_velocity,
    )
    priors = (
        (names[0], *defaults.narrow_velocity_bounds),
        (names[1], -13.0, -10.5),
        (names[2], 10.0, 120.0),
        (names[3], *defaults.broad_velocity_bounds),
        (names[4], 120.0, 500.0),
        (names[5], -2.0, 1.0),
        (names[6], 17.5, 18.9),
        (names[7], 8.6, 20.0),
        (names[8], *defaults.h1_velocity_bounds),
    )
    configuration = _configuration(names, base, priors)
    emission = CompositeEmissionModel(
        (
            GaussianEmissionComponent(LYA_REST_WAVELENGTH, 0, 2),
            GaussianEmissionComponent(
                LYA_REST_WAVELENGTH,
                3,
                4,
                log_weight_index=5,
            ),
        ),
        1,
    )
    absorption = CompositeAbsorptionModel((HydrogenDeuteriumAbsorber(6, 7, 8),))
    return make_model_catalog_entry(
        "double_gaussian",
        "Ordered narrow plus broad Gaussian emission without self-reversal",
        configuration,
        emission,
        operator,
        absorption_model=absorption,
        components=("Narrow Gaussian emission", "Broad Gaussian emission", "H I + D I"),
        tags=("two-component", "gaussian", "ordered-widths", "no-reversal"),
    )


def build_gj411_model_catalog(
    wavelength,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Return the curated GJ 411 catalog keyed by stable model names."""
    return build_lya_model_catalog(
        wavelength,
        defaults=GJ411_CATALOG_DEFAULTS,
        lsf_profile_column=3,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_e140m_target_model_catalog(
    target,
    wavelength,
    *,
    initial_log_total_flux=-11.75,
    oversample=8,
    model_padding=1.0,
):
    """Return the curated model set adapted to one E140M target."""
    defaults = catalog_defaults_for_target(
        target,
        initial_log_total_flux=initial_log_total_flux,
    )
    return build_lya_model_catalog(
        wavelength,
        defaults=defaults,
        lsf_profile_column=target.lsf_profile_column,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_lya_model_catalog(
    wavelength,
    *,
    defaults,
    lsf_profile_column=3,
    oversample=8,
    model_padding=1.0,
):
    """Return curated Ly-alpha models for explicit target and LSF defaults."""
    if not isinstance(defaults, LyaCatalogDefaults):
        raise TypeError("defaults must be LyaCatalogDefaults")
    operator = build_e140m_observation_operator(
        wavelength,
        lsf_profile_column=lsf_profile_column,
        oversample=oversample,
        model_padding=model_padding,
    )
    entries = (
        _single_gaussian(operator, defaults=defaults),
        _single_voigt(operator, defaults=defaults),
        _double_gaussian(operator, defaults=defaults),
        _single_gaussian(
            operator,
            gaussian_reversal=True,
            defaults=defaults,
        ),
        _single_voigt(operator, reversal_kind="legacy", defaults=defaults),
        _single_voigt(operator, reversal_kind="gaussian", defaults=defaults),
    )
    return {entry.name: entry for entry in entries}
