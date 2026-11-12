from pathlib import Path

import numpy as np
import pytest

from lyapy.models import (
    VoigtLyaParameters,
    evaluate_voigt_lya_model,
    legacy_voigt_model,
)
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


PACKAGE_DATA = Path(__file__).resolve().parents[1] / "src" / "lyapy"


def test_forward_model_components_are_physically_composed():
    reference_case = evaluate_gj411_legacy_case()
    result = evaluate_voigt_lya_model(
        reference_case["wave"],
        VoigtLyaParameters.from_sequence(PARAMETERS),
        reference_case["lsf_kernel"],
    )

    np.testing.assert_allclose(
        result.unconvolved.observed,
        result.unconvolved.intrinsic * result.unconvolved.attenuation,
        rtol=0.0,
        atol=0.0,
    )
    assert np.all((result.unconvolved.attenuation >= 0.0))
    assert np.all((result.unconvolved.attenuation <= 1.0))
    assert result.convolved.observed.shape == reference_case["wave"].shape


def test_legacy_adapter_matches_frozen_convolved_model(legacy_reference):
    wave = legacy_reference["wave"]
    raw_lsf = np.loadtxt(PACKAGE_DATA / "STIS_E140M_LSF.dat", skiprows=2)

    model = legacy_voigt_model(
        [wave],
        [[raw_lsf[:, 0], raw_lsf[:, 3]]],
        PARAMETERS,
        {"h1_vel": {"offset": False}},
    )

    np.testing.assert_allclose(
        model[0],
        legacy_reference["model_convolved"],
        rtol=1e-8,
        atol=1e-24,
    )


def test_parameter_sequence_requires_all_twelve_values():
    with pytest.raises(ValueError, match="expected 12 model parameters"):
        VoigtLyaParameters.from_sequence(PARAMETERS[:-1])


@pytest.mark.parametrize("bad_wave", [[1215.67], [[1215.6, 1215.7]], [1215.6, np.nan]])
def test_forward_model_rejects_invalid_wavelength_grids(bad_wave):
    with pytest.raises(ValueError, match="wavelength"):
        evaluate_voigt_lya_model(bad_wave, PARAMETERS)
