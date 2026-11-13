"""Differentiable JAX prototypes for Lyman-alpha forward models.

This module is optional: importing the rest of lyapy does not require JAX.
Enable JAX 64-bit calculations before calling these functions to preserve the
precision used by the NumPy/Astropy baseline::

    import jax
    jax.config.update("jax_enable_x64", True)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import voigt as legacy_voigt


LYA_REST_WAVELENGTH = 1215.67
LEGACY_SPEED_OF_LIGHT_KM_S = 3.0e5
CGS_SPEED_OF_LIGHT = 2.99792458e10
KM_SPEED_OF_LIGHT = 2.99792458e5
ELECTRON_CHARGE_ESU = 4.8032e-10
PROTON_MASS_G = 1.6726231e-24
ELECTRON_MASS_G = PROTON_MASS_G / 1836.0

_HUMLICEK_AA = np.array(
    [
        +46236.3358828121,
        -147726.58393079657j,
        -206562.80451354137,
        +281369.1590631087j,
        +183092.74968253175,
        -184787.96830696272j,
        -66155.39578477248,
        +57778.05827983565j,
        +11682.770904216826,
        -9442.402767960672j,
        -1052.8438624933142,
        +814.0996198624186j,
        +45.94499030751872,
        -34.59751573708725j,
        -0.7616559377907136,
        +0.5641895835476449j,
    ],
    dtype=np.complex128,
)
_HUMLICEK_BB = np.array(
    [
        +7918.06640624997,
        -126689.0625,
        +295607.8125,
        -236486.25,
        +84459.375,
        -15015.0,
        +1365.0,
        -60.0,
        +1.0,
    ],
    dtype=np.float64,
)
_LEGACY_U_VALUES = np.asarray(legacy_voigt.uvals)
_LEGACY_H1 = np.asarray(legacy_voigt.H1)
_LEGACY_H3 = np.asarray(legacy_voigt.H3)


class JaxLyaModelComponents(NamedTuple):
    """Physical model components represented as JAX arrays."""

    observed: jax.Array
    intrinsic: jax.Array
    attenuation: jax.Array
    reversal: jax.Array


class JaxLyaForwardModelResult(NamedTuple):
    """Unconvolved and instrument-convolved JAX model components."""

    unconvolved: JaxLyaModelComponents
    convolved: JaxLyaModelComponents | None
    stellar_voigt: jax.Array


def _require_x64():
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "lyapy's JAX model requires 64-bit mode; call "
            "jax.config.update('jax_enable_x64', True) before evaluation"
        )


def _humlicek_faddeeva(z):
    """JAX port of Astropy's Humlicek2 Faddeeva approximation."""
    coefficients = jnp.asarray(_HUMLICEK_AA)
    denominator_coefficients = jnp.asarray(_HUMLICEK_BB)
    zz = z * z
    exterior = (
        1j
        * z
        * (zz / jnp.sqrt(jnp.pi) - 1.410474)
        / (0.75 + zz * (zz - 3.0))
    )

    shifted_z = z + 1.35j
    shifted_zz = shifted_z * shifted_z
    numerator = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        numerator = numerator * shifted_z + coefficient
    denominator = jnp.ones_like(shifted_zz)
    for coefficient in denominator_coefficients[-2::-1]:
        denominator = denominator * shifted_zz + coefficient
    interior = numerator / denominator

    use_interior = jnp.abs(jnp.real(z)) + jnp.imag(z) < 10.0
    return jnp.where(use_interior, interior, exterior)


def _stellar_voigt(wavelength, line_center, amplitude, lorentz_fwhm, gaussian_fwhm):
    z = (
        (2.0 * (wavelength - line_center) + 1j * lorentz_fwhm)
        * jnp.sqrt(jnp.log(2.0))
        / gaussian_fwhm
    )
    return (
        jnp.real(_humlicek_faddeeva(z))
        * jnp.sqrt(jnp.log(2.0) * jnp.pi)
        / gaussian_fwhm
        * lorentz_fwhm
        * amplitude
    )


def stellar_voigt_profile_jax(
    wavelength,
    line_center,
    lorentz_fwhm,
    gaussian_fwhm,
):
    """Return a differentiable unit-amplitude Voigt profile in wavelength units."""
    _require_x64()
    return _stellar_voigt(
        jnp.asarray(wavelength, dtype=jnp.float64),
        line_center,
        1.0,
        lorentz_fwhm,
        gaussian_fwhm,
    )


def normalized_voigt_profile_jax(
    wavelength,
    line_center,
    lorentz_fwhm,
    gaussian_fwhm,
):
    """Return a Voigt probability density normalized over infinite wavelength."""
    _require_x64()
    wavelength = jnp.asarray(wavelength, dtype=jnp.float64)
    gaussian_sigma = gaussian_fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    lorentz_hwhm = 0.5 * lorentz_fwhm
    z = (
        (wavelength - line_center) + 1j * lorentz_hwhm
    ) / (gaussian_sigma * jnp.sqrt(2.0))
    return jnp.real(_humlicek_faddeeva(z)) / (
        gaussian_sigma * jnp.sqrt(2.0 * jnp.pi)
    )


def _harris_voigt(a_parameter, u_parameter):
    """JAX port of the Harris expansion used by the legacy absorption model."""
    u = jnp.abs(u_parameter)
    u_squared = u * u
    safe_u_squared = jnp.where(u_squared == 0.0, 1.0, u_squared)
    safe_u_fourth = safe_u_squared * safe_u_squared
    asymptotic = a_parameter / (jnp.sqrt(jnp.pi) * safe_u_squared) * (
        1.0
        + 1.5 / safe_u_squared
        + 3.75 / safe_u_fourth
        - a_parameter**2
        / safe_u_squared
        * (1.0 + 5.0 / safe_u_squared + 26.25 / safe_u_fourth)
    )

    exponential = jnp.exp(-u_squared)
    table_h1 = jnp.interp(u, jnp.asarray(_LEGACY_U_VALUES), jnp.asarray(_LEGACY_H1))
    table_h3 = jnp.interp(u, jnp.asarray(_LEGACY_U_VALUES), jnp.asarray(_LEGACY_H3))
    table = exponential + a_parameter * (
        table_h1
        + a_parameter
        * (
            (1.0 - 2.0 * u_squared) * exponential
            + a_parameter
            * (
                table_h3
                + a_parameter
                * (0.5 - 2.0 * u_squared + 2.0 / 3.0 * u_squared**2)
                * exponential
            )
        )
    )
    return jnp.where(u > 19.99, asymptotic, table)


def _line_optical_depth(
    wavelength,
    log_column,
    velocity,
    doppler_b,
    rest_wavelength,
    oscillator_strength,
    damping_rate,
):
    column_density = 10.0**log_column
    frequency_center = CGS_SPEED_OF_LIGHT / (rest_wavelength * 1.0e-8)
    doppler_frequency = frequency_center * doppler_b / KM_SPEED_OF_LIGHT
    a_parameter = jnp.abs(damping_rate / (4.0 * jnp.pi * doppler_frequency))
    cross_section_center = (
        jnp.sqrt(jnp.pi)
        * ELECTRON_CHARGE_ESU**2
        * oscillator_strength
        * rest_wavelength
        / (ELECTRON_MASS_G * CGS_SPEED_OF_LIGHT * doppler_b * 1.0e13)
    )

    shifted_center = (
        rest_wavelength + rest_wavelength * velocity / KM_SPEED_OF_LIGHT
    )
    wavelength_offset = jnp.minimum(jnp.abs(wavelength - shifted_center), 2.0)
    symmetric_blue_wavelength = rest_wavelength - wavelength_offset
    frequency = CGS_SPEED_OF_LIGHT / (symmetric_blue_wavelength * 1.0e-8)
    u_parameter = (frequency - frequency_center) / doppler_frequency
    cross_section = cross_section_center * _harris_voigt(a_parameter, u_parameter)
    return cross_section * column_density


def _hydrogen_deuterium_transmission(
    wavelength, log_h1_column, h1_doppler_b, h1_velocity, deuterium_to_hydrogen
):
    hydrogen_tau = _line_optical_depth(
        wavelength,
        log_h1_column,
        h1_velocity,
        h1_doppler_b,
        1215.67,
        0.4161,
        6.26e8,
    )
    deuterium_tau = _line_optical_depth(
        wavelength,
        log_h1_column + jnp.log10(deuterium_to_hydrogen),
        h1_velocity,
        h1_doppler_b / jnp.sqrt(2.0),
        1215.3394,
        0.4161,
        6.27e8,
    )
    return jnp.exp(-(hydrogen_tau + deuterium_tau))


def _convolve_components(components, kernel):
    return JaxLyaModelComponents(
        observed=jnp.convolve(components.observed, kernel, mode="same"),
        intrinsic=jnp.convolve(components.intrinsic, kernel, mode="same"),
        attenuation=jnp.convolve(components.attenuation, kernel, mode="same"),
        reversal=jnp.convolve(components.reversal, kernel, mode="same"),
    )


def evaluate_voigt_lya_model_jax(wavelength, parameters, convolution_kernel=None):
    """Evaluate the differentiable JAX equivalent of the NumPy baseline."""
    _require_x64()
    wave = jnp.asarray(wavelength, dtype=jnp.float64)
    parameter_vector = jnp.asarray(parameters, dtype=jnp.float64)
    (
        stellar_velocity,
        log_amplitude,
        lorentz_fwhm,
        gaussian_fwhm,
        h1_column,
        h1_doppler_b,
        h1_velocity,
        h1_column_2,
        h1_doppler_b_2,
        h1_velocity_2,
        deuterium_to_hydrogen,
        reversal_depth,
    ) = parameter_vector

    line_center = (
        stellar_velocity
        / LEGACY_SPEED_OF_LIGHT_KM_S
        * LYA_REST_WAVELENGTH
        + LYA_REST_WAVELENGTH
    )
    gaussian_fwhm_angstrom = (
        gaussian_fwhm / LEGACY_SPEED_OF_LIGHT_KM_S * LYA_REST_WAVELENGTH
    )
    lorentz_fwhm_angstrom = (
        lorentz_fwhm / LEGACY_SPEED_OF_LIGHT_KM_S * LYA_REST_WAVELENGTH
    )
    stellar_voigt = _stellar_voigt(
        wave,
        line_center,
        10.0**log_amplitude,
        lorentz_fwhm_angstrom,
        gaussian_fwhm_angstrom,
    )
    reversal = jnp.exp(-reversal_depth * stellar_voigt / jnp.max(stellar_voigt))
    intrinsic = stellar_voigt * reversal

    attenuation_1 = _hydrogen_deuterium_transmission(
        wave,
        h1_column,
        h1_doppler_b,
        h1_velocity,
        deuterium_to_hydrogen,
    )
    attenuation_2 = _hydrogen_deuterium_transmission(
        wave,
        h1_column_2,
        h1_doppler_b_2,
        h1_velocity_2,
        deuterium_to_hydrogen,
    )
    attenuation = attenuation_1 * attenuation_2
    unconvolved = JaxLyaModelComponents(
        observed=intrinsic * attenuation,
        intrinsic=intrinsic,
        attenuation=attenuation,
        reversal=reversal,
    )

    convolved = None
    if convolution_kernel is not None:
        kernel = jnp.asarray(convolution_kernel, dtype=jnp.float64)
        convolved = _convolve_components(unconvolved, kernel)

    return JaxLyaForwardModelResult(
        unconvolved=unconvolved,
        convolved=convolved,
        stellar_voigt=stellar_voigt,
    )
