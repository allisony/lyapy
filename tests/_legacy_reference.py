"""Frozen recipe for the legacy GJ 411 forward-model reference case.

This module deliberately mirrors the model assembled in
``example_start_file.py`` at Git commit ``bfb27ca``. It is test support, not a
new public API. Changes here should be reviewed as scientific changes.
"""

from pathlib import Path

import numpy as np
from scipy.io import readsav

from lyapy import lyapy
from lyapy.models import VoigtLyaParameters, evaluate_voigt_lya_model


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

    parameters = VoigtLyaParameters.from_sequence(PARAMETERS)
    model = evaluate_voigt_lya_model(wave, parameters, lsf_kernel)
    ism_component_1 = lyapy.total_tau_profile_func(
        wave,
        parameters.h1_column,
        parameters.h1_doppler_b,
        parameters.h1_velocity,
        parameters.deuterium_to_hydrogen,
    )
    ism_component_2 = lyapy.total_tau_profile_func(
        wave,
        parameters.h1_column_2,
        parameters.h1_doppler_b_2,
        parameters.h1_velocity_2,
        parameters.deuterium_to_hydrogen,
    )

    return {
        "parameter_names": PARAMETER_NAMES,
        "parameters": PARAMETERS,
        "continuum_coefficients": continuum_coefficients,
        "wave": wave,
        "flux": flux,
        "error": error,
        "lsf_kernel": lsf_kernel,
        "stellar_voigt": model.stellar_voigt,
        "reversal": model.unconvolved.reversal,
        "intrinsic": model.unconvolved.intrinsic,
        "ism_component_1": ism_component_1,
        "ism_component_2": ism_component_2,
        "ism_total": model.unconvolved.attenuation,
        "model_unconvolved": model.unconvolved.observed,
        "model_convolved": model.convolved.observed,
        "intrinsic_convolved": model.convolved.intrinsic,
        "integrated_intrinsic_flux": np.array(
            np.trapezoid(model.unconvolved.intrinsic, wave)
        ),
    }
