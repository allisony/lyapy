"""Interstellar absorption profiles for Lyman-alpha forward models."""

from types import MappingProxyType
import warnings

import numpy as np

from . import voigt
from .atomic_data import ATOMIC_TRANSITIONS, AtomicTransition


SPEED_OF_LIGHT_KM_S = 2.99792458e5
SPEED_OF_LIGHT_CM_S = 2.99792458e10
ELECTRON_CHARGE_ESU = 4.8032e-10
PROTON_MASS_G = 1.6726231e-24
ELECTRON_MASS_G = PROTON_MASS_G / 1836.0

_LEGACY_HYDROGEN_LYA = AtomicTransition(
    "legacy_h1",
    "H I",
    "Legacy H I Ly-alpha",
    1215.67,
    0.4161,
    6.26e8,
    "Legacy lyapy H I constants retained for numerical reproducibility",
)

_DEUTERIUM_LYA = AtomicTransition(
    "d_i_1215",
    "D I",
    "D I Ly-alpha",
    1215.3394,
    0.4161,
    6.27e8,
    "Legacy lyapy D I constants; not a Morton (2003) Table 2 entry",
)

# The historical ``mg2_h`` and ``mg2_k`` labels were reversed: 2796 A is k
# and 2803 A is h.  Preserve their numerical behavior as deprecated aliases,
# while exposing unambiguous canonical catalog keys for new calculations.
_LEGACY_LINE_ALIASES = MappingProxyType(
    {
        "h1": _LEGACY_HYDROGEN_LYA,
        "d1": _DEUTERIUM_LYA,
        "mg2_h": ATOMIC_TRANSITIONS["mg_ii_k"],
        "mg2_k": ATOMIC_TRANSITIONS["mg_ii_h"],
    }
)

LINE_DATA = MappingProxyType(
    {
        **{
            key: transition.absorption_parameters()
            for key, transition in ATOMIC_TRANSITIONS.items()
        },
        **{
            key: transition.absorption_parameters()
            for key, transition in _LEGACY_LINE_ALIASES.items()
        },
    }
)


def _absorption_parameters(line):
    if isinstance(line, AtomicTransition):
        return line.absorption_parameters()
    try:
        parameters = LINE_DATA[line]
    except (KeyError, TypeError) as error:
        choices = ", ".join(repr(name) for name in LINE_DATA)
        raise ValueError(
            f"line must be an AtomicTransition or one of {choices}"
        ) from error
    if line in ("mg2_h", "mg2_k"):
        canonical = "mg_ii_k" if line == "mg2_h" else "mg_ii_h"
        warnings.warn(
            f"{line!r} is a deprecated, historically reversed alias; "
            f"use {canonical!r}",
            DeprecationWarning,
            stacklevel=3,
        )
    return parameters


def tau_profile(
    log_column_density,
    velocity_shift,
    doppler_b,
    line,
    wave_cut_off=2.0,
):
    """Return the wavelength grid and optical depth of one absorber.

    This retains the reviewed legacy numerical grid so existing scientific
    results remain unchanged. Column density is log10(cm^-2); velocities and
    Doppler ``b`` are in km/s.
    """
    rest_wavelength, oscillator_strength, damping_rate = _absorption_parameters(
        line
    )

    column_density = 10.0**log_column_density
    grid_size = 4000
    central_frequency = SPEED_OF_LIGHT_CM_S / (rest_wavelength * 1e-8)
    doppler_frequency = central_frequency * doppler_b / SPEED_OF_LIGHT_KM_S
    damping_parameter = np.abs(
        damping_rate / (4.0 * np.pi * doppler_frequency)
    )
    near_center_cross_section = (
        np.sqrt(np.pi)
        * ELECTRON_CHARGE_ESU**2
        * oscillator_strength
        * rest_wavelength
        / (ELECTRON_MASS_G * SPEED_OF_LIGHT_CM_S * doppler_b * 1e13)
    )

    wave_edge = rest_wavelength - wave_cut_off
    wavelength_one_sided = np.linspace(rest_wavelength, wave_edge, num=grid_size)
    frequency_one_sided = SPEED_OF_LIGHT_CM_S / (wavelength_one_sided * 1e-8)
    u_parameter = (
        frequency_one_sided - central_frequency
    ) / doppler_frequency
    cross_section_one_sided = near_center_cross_section * voigt.voigt(
        damping_parameter,
        u_parameter,
    )
    cross_section_symmetric = np.append(
        cross_section_one_sided[::-1][0 : grid_size - 1],
        cross_section_one_sided,
    )

    first_point = wave_edge
    midpoint = rest_wavelength
    endpoint = 2.0 * (midpoint - first_point) + first_point
    wavelength_symmetric = np.linspace(
        first_point,
        endpoint,
        num=2 * grid_size - 1,
    )
    wavelength_shift = rest_wavelength * velocity_shift / SPEED_OF_LIGHT_KM_S
    wavelength_span = np.ptp(wavelength_symmetric)
    spacing = wavelength_symmetric[1] - wavelength_symmetric[0]
    output_size = np.round(wavelength_span / spacing)
    wavelength = (
        wavelength_span
        * (np.arange(output_size) / (output_size - 1))
        + wavelength_symmetric[0]
    )
    optical_depth = np.interp(
        wavelength,
        wavelength_symmetric + wavelength_shift,
        cross_section_symmetric * column_density,
    )
    return wavelength, optical_depth


def total_tau_profile(
    wavelength,
    h1_column,
    h1_doppler_b,
    h1_velocity,
    deuterium_to_hydrogen=1.5e-5,
    line="h1_d1",
    wave_cut_off=2.0,
):
    """Return the transmitted fraction for H I plus D I or one line."""
    if line == "h1_d1":
        h1_wave, h1_tau = tau_profile(
            h1_column,
            h1_velocity,
            h1_doppler_b,
            "h1",
            wave_cut_off=wave_cut_off,
        )
        h1_tau = np.interp(wavelength, h1_wave, h1_tau)
        d1_column = np.log10((10.0**h1_column) * deuterium_to_hydrogen)
        d1_wave, d1_tau = tau_profile(
            d1_column,
            h1_velocity,
            h1_doppler_b / np.sqrt(2.0),
            "d1",
            wave_cut_off=wave_cut_off,
        )
        d1_tau = np.interp(wavelength, d1_wave, d1_tau)
        optical_depth = h1_tau + d1_tau
    else:
        line_wave, optical_depth = tau_profile(
            h1_column,
            h1_velocity,
            h1_doppler_b,
            line,
            wave_cut_off=wave_cut_off,
        )
        optical_depth = np.interp(wavelength, line_wave, optical_depth)
    return np.exp(-optical_depth)


def total_tau_profile_func(
    wave_to_fit,
    h1_col,
    h1_b,
    h1_vel,
    d2h=1.5e-5,
    which_line="h1_d1",
    wave_cut_off=2.0,
):
    """Compatibility spelling and argument names for the historical API."""
    return total_tau_profile(
        wave_to_fit,
        h1_col,
        h1_b,
        h1_vel,
        deuterium_to_hydrogen=d2h,
        line=which_line,
        wave_cut_off=wave_cut_off,
    )
