"""Reviewed GJ 411 problem using the composable high-resolution model."""

from dataclasses import dataclass

import numpy as np

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.workflow import InferenceProblem
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    HydrogenDeuteriumAbsorber,
    LegacyIntensityReversal,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.observations import (
    GJ411_E140M_TARGET,
    ObservedSpectrum,
    StisTargetSpecification,
    TabulatedLsf,
    get_stis_instrument_configuration,
    load_stis_target_observation,
    load_stis_lsf,
)


COMPOSABLE_GJ411_PARAMETER_NAMES = (
    "stellar_velocity",
    "log_total_flux",
    "lorentz_fwhm",
    "gaussian_fwhm",
    "h1_column",
    "h1_doppler_b",
    "h1_velocity",
    "reversal_depth",
)

# The total flux matches the legacy GJ 411 baseline profile integrated over a
# sufficiently wide wavelength interval. Future inference fits this quantity
# directly rather than deriving it from an amplitude after sampling.
COMPOSABLE_GJ411_BASE_PARAMETERS = (
    -87.0,
    float(np.log10(1.7185359006885912e-12)),
    7.76,
    88.68,
    17.84,
    11.84,
    4.18,
    1.5,
)

GJ411_COMPOSABLE_CONFIGURATION = InferenceConfiguration(
    base_parameters=COMPOSABLE_GJ411_BASE_PARAMETERS,
    free_parameters=(
        UniformParameter("stellar_velocity", 0, -100.0, -75.0),
        UniformParameter("log_total_flux", 1, -13.0, -10.5),
        UniformParameter("lorentz_fwhm", 2, 1.0, 1000.0),
        UniformParameter("gaussian_fwhm", 3, 1.0, 1000.0),
        UniformParameter("h1_column", 4, 17.5, 18.9),
        UniformParameter("h1_doppler_b", 5, 8.6, 20.0),
        UniformParameter("h1_velocity", 6, -30.0, 30.0),
        UniformParameter("reversal_depth", 7, 0.0, 1.8),
    ),
    parameter_names=COMPOSABLE_GJ411_PARAMETER_NAMES,
)


@dataclass(frozen=True)
class ComposableInferenceProblem(InferenceProblem):
    """High-level inference problem with a composable forward model."""

    data: SpectrumData
    configuration: InferenceConfiguration
    forward_model: ComposableBinnedModel


def build_gj411_composable_forward_model(
    wavelength,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Build the reviewed stellar, ISM, and STIS operator for one grid."""
    operator = build_gj411_observation_operator(
        wavelength,
        oversample=oversample,
        model_padding=model_padding,
    )
    emission = CompositeEmissionModel(
        components=(VoigtEmissionComponent(1215.67, 0, 3, 2),),
        log_total_flux_index=1,
        reversal=LegacyIntensityReversal(7),
    )
    absorption = CompositeAbsorptionModel(
        components=(HydrogenDeuteriumAbsorber(4, 5, 6),),
    )
    return ComposableBinnedModel(
        emission,
        operator,
        absorption_model=absorption,
    )


def build_gj411_observation_operator(
    wavelength,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Build the shared E140M high-resolution operator for GJ 411."""
    return build_e140m_observation_operator(
        wavelength,
        lsf_profile_column=GJ411_E140M_TARGET.lsf_profile_column,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_e140m_observation_operator(
    wavelength,
    *,
    aperture="0.2x0.2",
    lsf_profile_column=None,
    lsf=None,
    oversample=8,
    model_padding=1.0,
):
    """Build an E140M operator from the bundled or supplied LSF."""
    return build_stis_observation_operator(
        wavelength,
        grating="E140M",
        aperture=aperture,
        lsf_profile_column=lsf_profile_column,
        lsf=lsf,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_g140m_observation_operator(
    wavelength,
    *,
    aperture="52x0.2",
    lsf_profile_column=None,
    lsf=None,
    oversample=8,
    model_padding=1.0,
):
    """Build a G140M operator from the bundled or supplied LSF."""
    return build_stis_observation_operator(
        wavelength,
        grating="G140M",
        aperture=aperture,
        lsf_profile_column=lsf_profile_column,
        lsf=lsf,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_g140l_observation_operator(
    wavelength,
    *,
    aperture="52x0.2",
    lsf_profile_column=None,
    lsf=None,
    oversample=8,
    model_padding=1.0,
):
    """Build a G140L operator from the bundled or supplied LSF."""
    return build_stis_observation_operator(
        wavelength,
        grating="G140L",
        aperture=aperture,
        lsf_profile_column=lsf_profile_column,
        lsf=lsf,
        oversample=oversample,
        model_padding=model_padding,
    )


def build_stis_observation_operator(
    wavelength,
    *,
    grating,
    aperture,
    lsf_profile_column=None,
    lsf=None,
    oversample=8,
    model_padding=1.0,
):
    """Build a high-resolution operator from a STIS pixel-space LSF."""
    get_stis_instrument_configuration(grating)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if lsf is None:
        lsf = load_stis_lsf(
            grating,
            aperture,
            wavelength=float(np.median(wavelength)),
            profile_column=lsf_profile_column,
        )
    elif not isinstance(lsf, TabulatedLsf):
        raise TypeError("lsf must be a TabulatedLsf")
    dispersion = float(np.median(np.diff(wavelength)))
    return build_high_resolution_operator(
        wavelength,
        lsf.offset * dispersion,
        lsf.profile,
        oversample=oversample,
        model_padding=model_padding,
    )


def load_stis_target_data(target):
    """Return inference data with target provenance and likelihood mask."""
    if not isinstance(target, StisTargetSpecification):
        raise TypeError("target must be a StisTargetSpecification")
    observation = load_stis_target_observation(target)
    return SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
        likelihood_mask=observation.likelihood_mask,
        metadata=observation.metadata,
    )


def prepare_observed_spectrum_data(
    spectrum,
    fit_window,
    *,
    exclusion_windows=(),
):
    """Crop raw flux without subtracting a continuum or airglow model.

    Quality-mask exclusions and optional wavelength exclusions are retained in
    the likelihood mask. Pixels with non-positive uncertainties remain on the
    native detector grid, but receive a finite placeholder uncertainty and are
    excluded from the likelihood. The placeholder never contributes to a fit.

    The convolution-kernel field is set to a unit impulse because composable
    models carry their high-resolution observation operator explicitly.
    """
    if not isinstance(spectrum, ObservedSpectrum):
        raise TypeError("spectrum must be an ObservedSpectrum")
    lower, upper = (float(value) for value in fit_window)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("fit_window bounds must be finite and increasing")
    fit_mask = (
        (spectrum.wavelength >= lower)
        & (spectrum.wavelength <= upper)
    )
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("fit_window must contain at least two detector pixels")
    wavelength = spectrum.wavelength[fit_mask]
    flux = spectrum.flux[fit_mask]
    error = spectrum.error[fit_mask].copy()
    likelihood_mask = spectrum.quality_mask[fit_mask] & (error > 0.0)
    parsed_exclusions = []
    for window in exclusion_windows:
        exclusion_lower, exclusion_upper = (float(value) for value in window)
        if (
            not np.isfinite(exclusion_lower)
            or not np.isfinite(exclusion_upper)
            or exclusion_lower >= exclusion_upper
        ):
            raise ValueError(
                "exclusion-window bounds must be finite and increasing"
            )
        likelihood_mask &= ~(
            (wavelength >= exclusion_lower)
            & (wavelength <= exclusion_upper)
        )
        parsed_exclusions.append((exclusion_lower, exclusion_upper))
    if not np.any(likelihood_mask):
        raise ValueError("no usable detector pixels remain in fit_window")
    invalid_error = error <= 0.0
    if np.any(invalid_error):
        error[invalid_error] = float(np.median(error[~invalid_error]))
    metadata = dict(spectrum.metadata)
    metadata.update(
        {
            "fit_window_angstrom": [lower, upper],
            "excluded_windows_angstrom": [
                list(window) for window in parsed_exclusions
            ],
            "continuum_subtracted": False,
            "airglow_subtracted": False,
            "nonpositive_error_pixels_excluded": int(
                np.count_nonzero(invalid_error)
            ),
        }
    )
    return SpectrumData(
        wavelength,
        flux,
        error,
        np.ones(1, dtype=np.float64),
        likelihood_mask=likelihood_mask,
        metadata=metadata,
    )


def load_e140m_target_data(target):
    """Backward-compatible alias for reviewed packaged E140M targets."""
    return load_stis_target_data(target)


def load_gj411_composable_problem(
    *,
    configuration=GJ411_COMPOSABLE_CONFIGURATION,
    oversample=8,
    model_padding=1.0,
):
    """Load observed GJ 411 data and its matched composable forward model."""
    data = load_e140m_target_data(GJ411_E140M_TARGET)
    forward_model = build_gj411_composable_forward_model(
        data.wavelength,
        oversample=oversample,
        model_padding=model_padding,
    )
    return ComposableInferenceProblem(
        data,
        configuration,
        forward_model,
        name="GJ 411 composable",
    )
