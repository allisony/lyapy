from dataclasses import replace
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lyapy.inference import (
    InferenceComparisonResult,
    InferenceComparisonRun,
    InferenceResult,
    ReconstructionSamples,
    SpectrumData,
    model_average,
    posterior_model_probabilities,
)


def make_result(name, log_evidence, log_evidence_error, predictions, weights):
    predictions = np.asarray(predictions, dtype=np.float64)
    wavelength = np.linspace(0.0, 2.0, predictions.shape[1])
    observation = SpectrumData(
        wavelength=wavelength,
        flux=np.full(wavelength.size, 5.0),
        error=np.full(wavelength.size, 0.5),
        convolution_kernel=np.ones(1),
    )
    integrated_flux = np.mean(predictions, axis=1)
    reconstruction = ReconstructionSamples(
        model_wavelength=wavelength,
        intrinsic=predictions,
        convolved_observed=predictions,
        integrated_intrinsic_flux=integrated_flux,
    )
    return InferenceResult(
        name=name,
        backend="dynesty",
        parameter_names=("amplitude",),
        samples=integrated_flux[:, None],
        weights=np.asarray(weights, dtype=np.float64),
        diagnostics={},
        problem_specification={"name": name},
        observations={"spectrum": observation},
        predictions={"spectrum": predictions},
        reconstructions={"spectrum": reconstruction},
        log_evidence=log_evidence,
        log_evidence_error=log_evidence_error,
    )


@pytest.fixture
def comparison():
    first = make_result(
        "model A",
        log_evidence=0.0,
        log_evidence_error=0.1,
        predictions=((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        weights=(0.25, 0.75),
    )
    second = make_result(
        "model B",
        log_evidence=math.log(3.0),
        log_evidence_error=0.2,
        predictions=((10.0, 10.0, 10.0), (12.0, 12.0, 12.0)),
        weights=(0.5, 0.5),
    )
    return InferenceComparisonResult(
        (
            InferenceComparisonRun(first, 1.0),
            InferenceComparisonRun(second, 2.0),
        )
    )


def test_equal_model_priors_convert_evidences_to_probabilities(comparison):
    probabilities = posterior_model_probabilities(comparison)
    averaged = comparison.model_average()

    assert probabilities == pytest.approx({"model A": 0.25, "model B": 0.75})
    assert averaged.model_prior_probabilities == {
        "model A": 0.5,
        "model B": 0.5,
    }
    assert averaged.model_probabilities == pytest.approx(probabilities)
    json.dumps(averaged.as_dict())


def test_explicit_model_priors_are_distinct_from_parameter_priors(comparison):
    averaged = model_average(
        comparison,
        {"model A": 0.75, "model B": 0.25},
    )

    assert averaged.model_probabilities == pytest.approx(
        {"model A": 0.5, "model B": 0.5}
    )


def test_model_average_combines_between_and_within_model_weights(comparison):
    averaged = comparison.model_average()
    predictions, weights = averaged.prediction_mixture()
    fluxes, flux_weights = averaged.integrated_flux_mixture()

    np.testing.assert_allclose(
        weights,
        (0.0625, 0.1875, 0.375, 0.375),
    )
    np.testing.assert_allclose(flux_weights, weights)
    np.testing.assert_allclose(predictions[:, 0], (0.0, 2.0, 10.0, 12.0))
    np.testing.assert_allclose(fluxes, predictions[:, 0])
    assert averaged.prediction_bands().shape == (5, 3)
    flux_summary = averaged.integrated_flux_summary()
    assert flux_summary["mean"] == pytest.approx(8.625)
    assert flux_summary["standard_deviation"] > 0.0


def test_evidence_error_probability_summary_is_reproducible(comparison):
    averaged = comparison.model_average()
    first = averaged.probability_summary(draws=1000, random_seed=17)
    repeated = averaged.probability_summary(draws=1000, random_seed=17)

    assert first == repeated
    assert first["model A"]["posterior_model_probability"] == pytest.approx(0.25)
    assert len(first["model A"]["probability_quantiles"]) == 3
    assert first["model A"]["probability_quantile_levels"] == [
        0.025,
        0.5,
        0.975,
    ]
    assert first["model A"]["probability_quantiles"][0] < 0.25
    assert first["model A"]["probability_quantiles"][2] > 0.25


def test_model_average_validates_priors_and_available_products(comparison):
    with pytest.raises(ValueError, match="missing model B"):
        comparison.model_average({"model A": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        comparison.model_average({"model A": -1.0, "model B": 2.0})
    with pytest.raises(KeyError, match="available spectra"):
        comparison.model_average().prediction_bands("missing")
    with pytest.raises(ValueError, match="no intrinsic reconstruction"):
        result = comparison.runs[0].result
        without_reconstruction = InferenceComparisonResult(
            (
                InferenceComparisonRun(
                    InferenceResult(
                        name=result.name,
                        backend=result.backend,
                        parameter_names=result.parameter_names,
                        samples=result.samples,
                        weights=result.weights,
                        diagnostics=result.diagnostics,
                        problem_specification=result.problem_specification,
                        observations=result.observations,
                        predictions=result.predictions,
                        log_evidence=result.log_evidence,
                        log_evidence_error=result.log_evidence_error,
                    ),
                    1.0,
                ),
                comparison.runs[1],
            )
        )
        without_reconstruction.model_average().integrated_flux_summary()

    changed_observation = SpectrumData(
        wavelength=np.linspace(0.0, 2.0, 3),
        flux=np.full(3, 6.0),
        error=np.full(3, 0.5),
        convolution_kernel=np.ones(1),
    )
    incompatible_result = replace(
        comparison.runs[1].result,
        observations={"spectrum": changed_observation},
    )
    incompatible = InferenceComparisonResult(
        (
            comparison.runs[0],
            InferenceComparisonRun(incompatible_result, 2.0),
        )
    )
    with pytest.raises(ValueError, match="identical observed spectra"):
        incompatible.model_average()


def test_model_average_plot_labels_component_probabilities(comparison):
    figure = comparison.model_average().plot()

    assert len(figure.axes) == 1
    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert "model A (25.0%)" in labels
    assert "model B (75.0%)" in labels
    plt.close(figure)
