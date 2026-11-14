"""Public scientific API for lyapy."""

from .absorption import tau_profile, total_tau_profile
from .atomic_data import (
    ATOMIC_TRANSITIONS,
    AtomicTransition,
    get_atomic_transition,
)
from .instrument import gaussian_kernel, interpolate_stis_lsf
from .likelihoods import (
    gaussian_log_likelihood,
    gaussian_pointwise_log_likelihood,
)
from .models import (
    LyaForwardModelResult,
    LyaModelComponents,
    VoigtLyaParameters,
    evaluate_voigt_lya_model,
)
from .observations import (
    AYRES_PHOTERR_SCALE,
    E140M_TARGETS,
    GJ411_E140M_TARGET,
    GJ729_E140M_TARGET,
    ObservedSpectrum,
    PreparedObservation,
    STIS_INSTRUMENT_CONFIGURATIONS,
    StisInstrumentConfiguration,
    StisLsfCalibration,
    StisTargetSpecification,
    TabulatedLsf,
    get_stis_instrument_configuration,
    load_gj411_observation,
    load_gj729_observation,
    load_stis_x1d,
    load_stis_lsf,
    load_stis_target_observation,
    prepare_stis_observation,
)
from .profiles import gaussian_flux, lya_intrinsic_profile
from .priors import legacy_bounded_log_prior


__all__ = [
    "ATOMIC_TRANSITIONS",
    "AYRES_PHOTERR_SCALE",
    "AtomicTransition",
    "E140M_TARGETS",
    "GJ411_E140M_TARGET",
    "GJ729_E140M_TARGET",
    "LyaForwardModelResult",
    "LyaModelComponents",
    "ObservedSpectrum",
    "PreparedObservation",
    "STIS_INSTRUMENT_CONFIGURATIONS",
    "StisInstrumentConfiguration",
    "StisLsfCalibration",
    "StisTargetSpecification",
    "TabulatedLsf",
    "VoigtLyaParameters",
    "evaluate_voigt_lya_model",
    "gaussian_flux",
    "gaussian_kernel",
    "gaussian_log_likelihood",
    "gaussian_pointwise_log_likelihood",
    "get_stis_instrument_configuration",
    "get_atomic_transition",
    "interpolate_stis_lsf",
    "lya_intrinsic_profile",
    "legacy_bounded_log_prior",
    "load_gj411_observation",
    "load_gj729_observation",
    "load_stis_x1d",
    "load_stis_lsf",
    "load_stis_target_observation",
    "prepare_stis_observation",
    "tau_profile",
    "total_tau_profile",
]
