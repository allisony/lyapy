import json

import numpy as np
import pytest

from lyapy.inference.common import (
    GJ411_CONFIGURATION,
    make_reversal_comparison_configurations,
)
from lyapy.inference.comparison import (
    EvidenceEstimate,
    ModelHypothesis,
    compare_evidence,
    evidence_strength,
    rank_evidence,
    run_evidence_comparison,
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


def test_reversal_comparison_configurations_share_common_priors():
    self_reversed, no_reversal = make_reversal_comparison_configurations(
        GJ411_CONFIGURATION
    )

    assert self_reversed.free_parameters[:-1] == no_reversal.free_parameters
    assert self_reversed.free_parameters[-1].name == "reversal_depth"
    assert self_reversed.free_parameters[-1].lower == 0.0
    assert self_reversed.free_parameters[-1].upper == 1.8
    assert no_reversal.base_parameters[11] == 0.0
    assert all(
        parameter.name != "reversal_depth"
        for parameter in no_reversal.free_parameters
    )


def test_reversal_comparison_requires_physical_varied_depth():
    _, no_reversal = make_reversal_comparison_configurations(GJ411_CONFIGURATION)

    with pytest.raises(ValueError, match="must vary"):
        make_reversal_comparison_configurations(no_reversal)
    with pytest.raises(ValueError, match="cannot be negative"):
        make_reversal_comparison_configurations(
            GJ411_CONFIGURATION,
            reversal_lower=-0.1,
        )


def test_evidence_ranking_supports_three_or_more_models():
    ranking = rank_evidence(
        (
            EvidenceEstimate("middle", 10.0, 0.2),
            EvidenceEstimate("best", 14.0, 0.3),
            EvidenceEstimate("last", 5.0, 0.4),
        )
    )

    assert [item.model for item in ranking] == ["best", "middle", "last"]
    assert [item.rank for item in ranking] == [1, 2, 3]
    assert ranking[0].delta_log_evidence_below_best == 0.0
    assert ranking[0].strength == "reference"
    np.testing.assert_allclose(ranking[1].delta_log_evidence_below_best, 4.0)
    assert ranking[2].strength == "strong"


def test_generic_evidence_runner_retains_samplers_and_serializes_report():
    class Results:
        def __init__(self, log_evidence):
            self.logz = np.array([log_evidence])
            self.logzerr = np.array([0.2])

    class Sampler:
        def __init__(self, log_evidence):
            self.results = Results(log_evidence)

    def fake_runner(
        data,
        configuration,
        random_state,
        forward_model,
        likelihood_function,
        **options,
    ):
        assert data == "spectrum"
        assert options == {"nlive": 25, "print_progress": False}
        assert random_state is not None
        assert callable(forward_model)
        assert callable(likelihood_function)
        return Sampler(float(configuration.ndim))

    self_reversed, no_reversal = make_reversal_comparison_configurations(
        GJ411_CONFIGURATION
    )
    hypotheses = (
        ModelHypothesis("self reversal", self_reversed),
        ModelHypothesis("no reversal", no_reversal),
    )
    result = run_evidence_comparison(
        "spectrum",
        hypotheses,
        random_seed=42,
        sampler_runner=fake_runner,
        nlive=25,
        print_progress=False,
    )

    assert result.best_model == "self reversal"
    assert result.get_run("no reversal").sampler.results.logz[0] == 7.0
    assert result.get_run("self reversal").sampler.results.logz[0] == 8.0
    report = result.as_dict()
    assert report["models"]["self reversal"]["hypothesis"][
        "forward_model"
    ].endswith("evaluate_voigt_lya_model_jax")
    assert report["models"]["self reversal"]["hypothesis"][
        "likelihood_function"
    ].endswith("gaussian_log_likelihood")
    json.dumps(report)


def test_generic_evidence_runner_validates_hypothesis_collection():
    hypothesis = ModelHypothesis("same", GJ411_CONFIGURATION)

    with pytest.raises(ValueError, match="at least two"):
        run_evidence_comparison(None, (hypothesis,))
    with pytest.raises(ValueError, match="unique"):
        run_evidence_comparison(None, (hypothesis, hypothesis))
