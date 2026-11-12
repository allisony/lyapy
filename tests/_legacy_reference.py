"""Frozen recipe for the legacy GJ 411 forward-model reference case.

This module deliberately mirrors the model assembled in
``example_start_file.py`` at Git commit ``bfb27ca``. It is test support, not a
new public API. Changes here should be reviewed as scientific changes.
"""

from pathlib import Path

import numpy as np
from astropy.modeling.models import Voigt1D
from scipy.io import readsav

from lyapy import lyapy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA = REPOSITORY_ROOT / "src" / "lyapy"

PARAMETER_NAMES = np.array(
    [
        "vs",
        "am",
        "fw_L",
        "fw_G",
        "h1_col",
        "h1_b",
        "h1_vel",
        "h1_col2",
        "h1_b2",
        "h1_vel2",
        "d2h",
        "p",
    ]
)

PARAMETERS = np.array(
    [
        -87.0,
        -10.06,
        7.76,
        88.68,
        17.84,
        11.84,
        4.18,
        0.0,
        11.5,
        5.87,
        1.5e-5,
        1.5,
    ]
)


def evaluate_gj411_legacy_case():
    """Evaluate the deterministic legacy GJ 411 preprocessing and model."""
    spectrum = readsav(PACKAGE_DATA / "HD-95735-E140M-odn905010_1.xl_stis")
    full_wave = spectrum["wave"]
    full_flux = spectrum["flux"]
    full_error = spectrum["photerr"]

    continuum_mask = ((full_wave > 1212.0) & (full_wave < 1213.0)) | (
        (full_wave > 1217.0) & (full_wave < 1217.9)
    )
    continuum_coefficients = np.polyfit(
        full_wave[continuum_mask], full_flux[continuum_mask], deg=1
    )
    continuum = np.polyval(continuum_coefficients, full_wave)

    fit_mask = (full_wave >= 1214.5) & (full_wave <= 1216.55)
    wave = full_wave[fit_mask]
    flux = full_flux[fit_mask] - continuum[fit_mask]
    error = full_error[fit_mask]

    raw_lsf = np.loadtxt(PACKAGE_DATA / "STIS_E140M_LSF.dat", skiprows=2)
    lsf_kernel = lyapy.ready_stis_lsf(
        raw_lsf[:, 0], raw_lsf[:, 3], wave[1] - wave[0], wave
    )

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
    ) = PARAMETERS

    line_center = stellar_velocity / 3.0e5 * 1215.67 + 1215.67
    gaussian_fwhm_angstrom = gaussian_fwhm / 3.0e5 * 1215.67
    lorentz_fwhm_angstrom = lorentz_fwhm / 3.0e5 * 1215.67
    voigt_model = Voigt1D(
        x_0=line_center,
        amplitude_L=10.0**log_amplitude,
        fwhm_L=lorentz_fwhm_angstrom,
        fwhm_G=gaussian_fwhm_angstrom,
    )
    stellar_voigt = voigt_model(wave)
    reversal = np.exp(
        -reversal_depth * stellar_voigt / np.max(stellar_voigt)
    )
    intrinsic = stellar_voigt * reversal

    ism_component_1 = lyapy.total_tau_profile_func(
        wave,
        h1_column,
        h1_doppler_b,
        h1_velocity,
        deuterium_to_hydrogen,
    )
    ism_component_2 = lyapy.total_tau_profile_func(
        wave,
        h1_column_2,
        h1_doppler_b_2,
        h1_velocity_2,
        deuterium_to_hydrogen,
    )
    ism_total = ism_component_1 * ism_component_2
    model_unconvolved = intrinsic * ism_total

    return {
        "parameter_names": PARAMETER_NAMES,
        "parameters": PARAMETERS,
        "continuum_coefficients": continuum_coefficients,
        "wave": wave,
        "flux": flux,
        "error": error,
        "lsf_kernel": lsf_kernel,
        "stellar_voigt": stellar_voigt,
        "reversal": reversal,
        "intrinsic": intrinsic,
        "ism_component_1": ism_component_1,
        "ism_component_2": ism_component_2,
        "ism_total": ism_total,
        "model_unconvolved": model_unconvolved,
        "model_convolved": np.convolve(model_unconvolved, lsf_kernel, mode="same"),
        "intrinsic_convolved": np.convolve(intrinsic, lsf_kernel, mode="same"),
        "integrated_intrinsic_flux": np.array(np.trapezoid(intrinsic, wave)),
    }
