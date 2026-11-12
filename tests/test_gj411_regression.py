import numpy as np
import pytest

from tests._legacy_reference import evaluate_gj411_legacy_case


REFERENCE_ARRAYS = [
    ("continuum_coefficients", 1e-24),
    ("wave", 1e-12),
    ("flux", 1e-24),
    ("error", 1e-24),
    ("lsf_kernel", 1e-14),
    ("stellar_voigt", 1e-24),
    ("reversal", 1e-14),
    ("intrinsic", 1e-24),
    ("ism_component_1", 1e-14),
    ("ism_component_2", 1e-14),
    ("ism_total", 1e-14),
    ("model_unconvolved", 1e-24),
    ("model_convolved", 1e-24),
    ("intrinsic_convolved", 1e-24),
]


@pytest.fixture(scope="module")
def current_legacy_evaluation():
    return evaluate_gj411_legacy_case()


@pytest.mark.parametrize("array_name, absolute_tolerance", REFERENCE_ARRAYS)
def test_gj411_legacy_array_matches_reviewed_reference(
    array_name, absolute_tolerance, current_legacy_evaluation, legacy_reference
):
    np.testing.assert_allclose(
        current_legacy_evaluation[array_name],
        legacy_reference[array_name],
        rtol=1e-8,
        atol=absolute_tolerance,
    )


def test_gj411_parameter_vector_matches_reviewed_reference(
    current_legacy_evaluation, legacy_reference
):
    np.testing.assert_array_equal(
        current_legacy_evaluation["parameter_names"],
        legacy_reference["parameter_names"],
    )
    np.testing.assert_allclose(
        current_legacy_evaluation["parameters"],
        legacy_reference["parameters"],
        rtol=0.0,
        atol=0.0,
    )


def test_gj411_integrated_intrinsic_flux_matches_reviewed_reference(
    current_legacy_evaluation, legacy_reference
):
    np.testing.assert_allclose(
        current_legacy_evaluation["integrated_intrinsic_flux"],
        legacy_reference["integrated_intrinsic_flux"],
        rtol=1e-8,
        atol=0.0,
    )
