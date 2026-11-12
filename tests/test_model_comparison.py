import numpy as np
import pytest

from lyapy.inference.comparison import (
    EvidenceEstimate,
    compare_evidence,
    evidence_strength,
)


def test_evidence_comparison_reports_direction_uncertainty_and_scale():
    reversal = EvidenceEstimate("self reversal", 102.0, 0.3)
    no_reversal = EvidenceEstimate("no reversal", 96.0, 0.4)

    comparison = compare_evidence(reversal, no_reversal)

    assert comparison.favored_model == "self reversal"
    assert comparison.strength == "strong"
    np.testing.assert_allclose(comparison.delta_log_evidence, 6.0)
    np.testing.assert_allclose(comparison.delta_log_evidence_error, 0.5)
    np.testing.assert_allclose(
        comparison.log10_bayes_factor,
        6.0 / np.log(10.0),
    )
    assert comparison.as_dict()["numerator"]["model"] == "self reversal"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.5, "inconclusive"), (1.5, "weak"), (3.0, "moderate"), (6.0, "strong")],
)
def test_evidence_strength_categories_are_symmetric(delta, expected):
    assert evidence_strength(delta) == expected
    assert evidence_strength(-delta) == expected


def test_evidence_estimate_validates_inputs():
    with pytest.raises(ValueError, match="name"):
        EvidenceEstimate("", 1.0, 0.1)
    with pytest.raises(ValueError, match="error"):
        EvidenceEstimate("model", 1.0, -0.1)
