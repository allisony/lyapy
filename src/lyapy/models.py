"""Importable forward models for Lyman-alpha reconstruction.

The functions in this module contain no plotting, file I/O, or sampler state.
They provide a stable NumPy baseline for the later differentiable model work.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from astropy.modeling.models import Voigt1D

from .absorption import total_tau_profile
from .instrument import interpolate_stis_lsf


LYA_REST_WAVELENGTH = 1215.67
LEGACY_SPEED_OF_LIGHT_KM_S = 3.0e5


@dataclass(frozen=True)
class VoigtLyaParameters:
    """Parameters for the legacy two-absorber, self-reversed Voigt model."""

    stellar_velocity: float
    log_amplitude: float
    lorentz_fwhm: float
    gaussian_fwhm: float
    h1_column: float
    h1_doppler_b: float
    h1_velocity: float
    h1_column_2: float
    h1_doppler_b_2: float
    h1_velocity_2: float
    deuterium_to_hydrogen: float
    reversal_depth: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "VoigtLyaParameters":
        """Build parameters from the ordering used by the legacy fitter."""
        values = tuple(values)
        if len(values) != 12:
            raise ValueError(f"expected 12 model parameters, received {len(values)}")
        return cls(*values)


@dataclass(frozen=True)
class LyaModelComponents:
    """Physical components evaluated on a common wavelength grid."""

    observed: np.ndarray
    intrinsic: np.ndarray
    attenuation: np.ndarray
    reversal: np.ndarray


@dataclass(frozen=True)
class LyaForwardModelResult:
    """Unconvolved components and, when requested, convolved components."""

    unconvolved: LyaModelComponents
    convolved: LyaModelComponents | None
    stellar_voigt: np.ndarray


def _validate_vector(values, name, minimum_size=1):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _convolve_components(components, kernel):
    return LyaModelComponents(
        observed=np.convolve(components.observed, kernel, mode="same"),
        intrinsic=np.convolve(components.intrinsic, kernel, mode="same"),
        attenuation=np.convolve(components.attenuation, kernel, mode="same"),
        reversal=np.convolve(components.reversal, kernel, mode="same"),
    )


def evaluate_voigt_lya_model(wavelength, parameters, convolution_kernel=None):
    """Evaluate the legacy GJ 411-style Lyman-alpha forward model.

    Parameters are velocities and widths in km/s, logarithmic amplitude and
    H I columns in base-10 units, and a dimensionless self-reversal depth.
    Passing a convolution kernel adds instrument-convolved components while
    retaining the unconvolved physical model in the result.
    """
    wave = _validate_vector(wavelength, "wavelength", minimum_size=2)
    if not isinstance(parameters, VoigtLyaParameters):
        parameters = VoigtLyaParameters.from_sequence(parameters)

    line_center = (
        parameters.stellar_velocity
        / LEGACY_SPEED_OF_LIGHT_KM_S
        * LYA_REST_WAVELENGTH
        + LYA_REST_WAVELENGTH
    )
    gaussian_fwhm_angstrom = (
        parameters.gaussian_fwhm
        / LEGACY_SPEED_OF_LIGHT_KM_S
        * LYA_REST_WAVELENGTH
    )
    lorentz_fwhm_angstrom = (
        parameters.lorentz_fwhm
        / LEGACY_SPEED_OF_LIGHT_KM_S
        * LYA_REST_WAVELENGTH
    )
    voigt_model = Voigt1D(
        x_0=line_center,
        amplitude_L=10.0**parameters.log_amplitude,
        fwhm_L=lorentz_fwhm_angstrom,
        fwhm_G=gaussian_fwhm_angstrom,
    )
    stellar_voigt = np.asarray(voigt_model(wave))
    reversal = np.exp(
        -parameters.reversal_depth * stellar_voigt / np.max(stellar_voigt)
    )
    intrinsic = stellar_voigt * reversal

    attenuation_1 = total_tau_profile(
        wave,
        parameters.h1_column,
        parameters.h1_doppler_b,
        parameters.h1_velocity,
        deuterium_to_hydrogen=parameters.deuterium_to_hydrogen,
    )
    attenuation_2 = total_tau_profile(
        wave,
        parameters.h1_column_2,
        parameters.h1_doppler_b_2,
        parameters.h1_velocity_2,
        deuterium_to_hydrogen=parameters.deuterium_to_hydrogen,
    )
    attenuation = attenuation_1 * attenuation_2
    unconvolved = LyaModelComponents(
        observed=intrinsic * attenuation,
        intrinsic=intrinsic,
        attenuation=attenuation,
        reversal=reversal,
    )

    convolved = None
    if convolution_kernel is not None:
        kernel = _validate_vector(convolution_kernel, "convolution_kernel")
        convolved = _convolve_components(unconvolved, kernel)

    return LyaForwardModelResult(
        unconvolved=unconvolved,
        convolved=convolved,
        stellar_voigt=stellar_voigt,
    )


def legacy_voigt_model(
    x,
    resolution,
    parameters,
    variables: Mapping,
    lnlike=True,
    convolve=True,
):
    """Compatibility adapter for the callable expected by ``fitting.py``."""
    wave = _validate_vector(x[0], "wavelength", minimum_size=2)
    raw_lsf_wave, raw_lsf = resolution[0]
    kernel = interpolate_stis_lsf(
        raw_lsf_wave,
        raw_lsf,
        wave[1] - wave[0],
        wave,
    )
    model_parameters = VoigtLyaParameters.from_sequence(parameters)
    if variables["h1_vel"].get("offset", False):
        model_parameters = replace(
            model_parameters,
            h1_velocity=(
                model_parameters.h1_velocity + model_parameters.stellar_velocity
            ),
        )

    result = evaluate_voigt_lya_model(wave, model_parameters, kernel)
    if lnlike:
        if not convolve:
            raise ValueError("convolve=False when it should be True")
        return [result.convolved.observed]

    if convolve == "both":
        return [
            [result.convolved.observed],
            [result.unconvolved.observed],
            [result.convolved.intrinsic],
            [result.unconvolved.intrinsic],
            [result.convolved.attenuation],
            [result.unconvolved.attenuation],
            [result.convolved.reversal],
            [result.unconvolved.reversal],
        ]

    components = result.convolved if convolve else result.unconvolved
    return [
        [components.observed],
        [components.intrinsic],
        [components.attenuation],
        [components.reversal],
    ]
