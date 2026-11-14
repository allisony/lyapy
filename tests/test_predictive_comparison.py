from dataclasses import replace
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest


pytest.importorskip("arviz")

from lyapy.inference import (
    InferenceResult,
    SpectrumData,
    compare_nuts_results,
    evaluate_psis_loo,
)
from lyapy.inference.reconstruction import ReconstructionSamples
from lyapy.likelihoods import gaussian_pointwise_log_likelihood


def make_result(name, *, bias=0.0):
    rng = np.random.default_rng(71)
    wavelength = np.linspace(1215.1, 1216.1, 12)
    truth = np.exp(-0.5 * ((wavelength - 1215.65) / 0.18) ** 2)
    error = np.full(wavelength.size, 0.12)
    observed = truth + rng.normal(0.0, error)
    data = SpectrumData(wavelength, observed, error, np.ones(1))
    draws = 400
    offsets = rng.normal(bias, 0.035, draws)
    predictions = truth[None, :] + offsets[:, None]
    pointwise = np.vstack(
        [
            gaussian_pointwise_log_likelihood(observed, row, error)
            for row in predictions
        ]
    )
    intrinsic = np.repeat(truth[None, :], draws, axis=0)
    integrated_flux = np.trapezoid(intrinsic, wavelength, axis=1) + offsets
    reconstruction = ReconstructionSamples(
        model_wavelength=wavelength,
        intrinsic=intrinsic,
        convolved_observed=predictions,
        integrated_intrinsic_flux=integrated_flux,
    )
    return InferenceResult(
        name=name,
        backend="nuts",
        parameter_names=("offset",),
        samples=offsets[:, None],
        weights=None,
        diagnostics={"chains": 2, "draws_per_chain": 200},
        problem_specification={"likelihood": "Gaussian"},
        observations={"spectrum": data},
        predictions={"spectrum": predictions},
        reconstructions={"spectrum": reconstruction},
        pointwise_log_likelihoods={"spectrum": pointwise},
    )


@pytest.fixture(scope="module")
def comparison():
    return compare_nuts_results(
        (make_result("well specified"), make_result("biased", bias=0.35))
    )


def test_psis_loo_reports_pixel_level_diagnostics(comparison):
    result = comparison.get_loo_result("well specified")

    assert result.pointwise_elpd.shape == (12,)
    assert result.pareto_k.shape == (12,)
    assert result.wavelength.shape == (12,)
    assert np.isfinite(result.elpd_loo)
    assert result.good_k_threshold > 0.0
    assert result.loo_information_criterion == pytest.approx(
        -2.0 * result.elpd_loo
    )
    direct = evaluate_psis_loo(comparison.get_result("well specified"))
    assert direct.elpd_loo == pytest.approx(result.elpd_loo)


def test_comparison_ranks_models_and_reports_paired_difference(comparison):
    assert comparison.best_model == "well specified"
    assert comparison.ranking[0].model == "well specified"
    assert sum(comparison.stacking_weights.values()) == pytest.approx(1.0)
    assert comparison.stacking_weights["well specified"] > 0.9

    difference = comparison.compare("well specified", "biased")
    assert difference.delta_elpd > 0.0
    assert difference.standard_error >= 0.0
    assert difference.favored_model == "well specified"


def test_stacking_combines_predictions_and_intrinsic_flux(comparison):
    samples, weights = comparison.prediction_mixture()
    bands = comparison.prediction_bands()
    flux = comparison.integrated_flux_summary()

    assert samples.shape == (800, 12)
    assert weights.shape == (800,)
    assert np.sum(weights) == pytest.approx(1.0)
    assert bands.shape == (5, 12)
    expected_mean = sum(
        comparison.stacking_weights[result.name]
        * np.mean(
            result.reconstructions["spectrum"].integrated_intrinsic_flux
        )
        for result in comparison.results
    )
    assert flux["mean"] == pytest.approx(expected_mean)


def test_comparison_serializes_and_plots(comparison, tmp_path):
    path = comparison.save(tmp_path / "psis-comparison")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["best_predictive_model"] == "well specified"
    assert "optimistic" in report["caveat"]
    json.dumps(report, allow_nan=False)
    figures = (
        comparison.plot(),
        comparison.plot_average(),
        comparison.loo_results[0].plot_pareto_k(),
    )
    assert [len(figure.axes) for figure in figures] == [2, 1, 1]
    for figure in figures:
        plt.close(figure)


def test_infinite_pareto_k_is_visible_and_json_safe(comparison):
    loo = comparison.loo_results[0]
    infinite = replace(loo, pareto_k=np.full(loo.pareto_k.shape, np.inf))

    report = infinite.as_dict()
    assert report["maximum_pareto_k"] is None
    assert report["infinite_pareto_k"] == loo.pareto_k.size
    assert all(value is None for value in report["pareto_k"])
    json.dumps(report, allow_nan=False)
    figure = infinite.plot_pareto_k()
    assert "infinite k" in " ".join(
        text.get_text() for text in figure.axes[0].get_legend().get_texts()
    )
    assert figure.axes[0].collections[0].get_offsets().shape[0] == 0
    assert figure.axes[0].collections[1].get_offsets().shape[0] == (
        loo.pareto_k.size
    )
    plt.close(figure)


def test_comparison_rejects_incompatible_results():
    first = make_result("first")
    second = make_result("second")
    with pytest.raises(ValueError, match="at least two"):
        compare_nuts_results((first,))
    with pytest.raises(ValueError, match="pointwise log likelihood"):
        evaluate_psis_loo(replace(first, pointwise_log_likelihoods={}))
    changed_observation = replace(
        second.observations["spectrum"],
        flux=second.observations["spectrum"].flux + 0.01,
    )
    with pytest.raises(ValueError, match="identical observations"):
        compare_nuts_results(
            (
                first,
                replace(
                    second,
                    observations={"spectrum": changed_observation},
                ),
            )
        )
    dynesty = replace(
        first,
        backend="dynesty",
        weights=np.full(first.draws, 1.0 / first.draws),
        log_evidence=0.0,
        log_evidence_error=0.1,
    )
    with pytest.raises(ValueError, match="NUTS"):
        evaluate_psis_loo(dynesty)
