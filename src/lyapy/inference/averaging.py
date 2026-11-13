"""Evidence-weighted Bayesian model averaging for portable comparisons."""

from dataclasses import dataclass
import math

import numpy as np

from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.inference.workflow import InferenceComparisonResult


def _normalized_probabilities(values, names, description):
    values = dict(values)
    expected = set(names)
    supplied = set(values)
    if supplied != expected:
        missing = expected - supplied
        unknown = supplied - expected
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(f"{description} must match all models: {'; '.join(details)}")
    probabilities = np.asarray([values[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{description} must be finite and non-negative")
    total = np.sum(probabilities)
    if total <= 0.0:
        raise ValueError(f"{description} must have a positive sum")
    probabilities = probabilities / total
    return {
        name: float(probability)
        for name, probability in zip(names, probabilities)
    }


def posterior_model_probabilities(
    comparison,
    model_prior_probabilities=None,
):
    """Return model probabilities implied by evidences and model priors.

    Equal prior probability is assigned to each model when
    ``model_prior_probabilities`` is omitted. These hypothesis-level prior
    probabilities are distinct from the parameter priors used to calculate
    each model's evidence.
    """
    if not isinstance(comparison, InferenceComparisonResult):
        raise TypeError("comparison must be an InferenceComparisonResult")
    names = tuple(run.result.name for run in comparison.runs)
    if model_prior_probabilities is None:
        model_prior_probabilities = {name: 1.0 for name in names}
    priors = _normalized_probabilities(
        model_prior_probabilities,
        names,
        "model_prior_probabilities",
    )
    log_values = np.asarray(
        [
            run.result.log_evidence
            + (
                math.log(priors[run.result.name])
                if priors[run.result.name] > 0.0
                else -np.inf
            )
            for run in comparison.runs
        ]
    )
    maximum = np.max(log_values)
    probabilities = np.exp(log_values - maximum)
    probabilities = probabilities / np.sum(probabilities)
    return {
        name: float(probability)
        for name, probability in zip(names, probabilities)
    }


def _result_weights(result):
    if result.weights is None:
        return np.full(result.draws, 1.0 / result.draws)
    return result.weights / np.sum(result.weights)


def _same_observations(first, second):
    if tuple(first) != tuple(second):
        return False
    return all(
        all(
            np.array_equal(
                getattr(first[name], attribute),
                getattr(second[name], attribute),
            )
            for attribute in (
                "wavelength",
                "flux",
                "error",
                "convolution_kernel",
            )
        )
        for name in first
    )


@dataclass(frozen=True)
class ModelAveragedResult:
    """Posterior products mixed across models using evidence weights."""

    comparison: InferenceComparisonResult
    model_prior_probabilities: dict[str, float]
    model_probabilities: dict[str, float]

    def __post_init__(self):
        if not isinstance(self.comparison, InferenceComparisonResult):
            raise TypeError("comparison must be an InferenceComparisonResult")
        names = tuple(run.result.name for run in self.comparison.runs)
        priors = _normalized_probabilities(
            self.model_prior_probabilities,
            names,
            "model_prior_probabilities",
        )
        probabilities = _normalized_probabilities(
            self.model_probabilities,
            names,
            "model_probabilities",
        )
        expected = posterior_model_probabilities(self.comparison, priors)
        if not np.allclose(
            [probabilities[name] for name in names],
            [expected[name] for name in names],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "model_probabilities are inconsistent with evidences and priors"
            )
        reference_observations = self.comparison.runs[0].result.observations
        if not all(
            _same_observations(
                reference_observations,
                run.result.observations,
            )
            for run in self.comparison.runs[1:]
        ):
            raise ValueError(
                "model averaging requires identical observed spectra"
            )
        object.__setattr__(self, "model_prior_probabilities", priors)
        object.__setattr__(self, "model_probabilities", probabilities)

    @property
    def model_names(self):
        return tuple(run.result.name for run in self.comparison.runs)

    def probability_summary(
        self,
        *,
        draws=10000,
        random_seed=0,
        probabilities=(0.025, 0.5, 0.975),
    ):
        """Approximate uncertainty from reported log-evidence errors.

        This treats the reported log-evidence estimates as independent normal
        variables. The central probabilities used for model averaging are not
        randomly perturbed.
        """
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive integer")
        if not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        quantiles = np.asarray(probabilities, dtype=np.float64)
        if quantiles.ndim != 1 or quantiles.size == 0:
            raise ValueError("probabilities must be a non-empty vector")
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("probabilities must lie between zero and one")
        random = np.random.default_rng(random_seed)
        means = np.asarray(
            [run.result.log_evidence for run in self.comparison.runs]
        )
        errors = np.asarray(
            [run.result.log_evidence_error for run in self.comparison.runs]
        )
        log_priors = np.asarray(
            [
                math.log(self.model_prior_probabilities[name])
                if self.model_prior_probabilities[name] > 0.0
                else -np.inf
                for name in self.model_names
            ]
        )
        sampled_log_weights = random.normal(means, errors, size=(draws, len(means)))
        sampled_log_weights = sampled_log_weights + log_priors
        sampled_log_weights = sampled_log_weights - np.max(
            sampled_log_weights,
            axis=1,
            keepdims=True,
        )
        sampled_probabilities = np.exp(sampled_log_weights)
        sampled_probabilities = sampled_probabilities / np.sum(
            sampled_probabilities,
            axis=1,
            keepdims=True,
        )
        return {
            name: {
                "model_prior_probability": self.model_prior_probabilities[name],
                "posterior_model_probability": self.model_probabilities[name],
                "log_evidence_error": float(errors[index]),
                "probability_quantile_levels": quantiles.tolist(),
                "probability_quantiles": np.quantile(
                    sampled_probabilities[:, index],
                    quantiles,
                ).tolist(),
            }
            for index, name in enumerate(self.model_names)
        }

    def prediction_mixture(self, spectrum="spectrum"):
        """Return detector predictions and their normalized mixture weights."""
        samples = []
        weights = []
        for run in self.comparison.runs:
            result = run.result
            try:
                prediction = result.predictions[spectrum]
            except KeyError as error:
                available = ", ".join(result.predictions)
                raise KeyError(
                    f"unknown spectrum {spectrum!r}; available spectra: {available}"
                ) from error
            samples.append(prediction)
            weights.append(
                self.model_probabilities[result.name] * _result_weights(result)
            )
        return np.concatenate(samples), np.concatenate(weights)

    def prediction_bands(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        """Summarize the evidence-weighted detector-space mixture."""
        predictions, weights = self.prediction_mixture(spectrum)
        return posterior_percentile_bands(
            predictions,
            probabilities,
            weights=weights,
        )

    def integrated_flux_mixture(self, spectrum="spectrum"):
        """Return intrinsic integrated fluxes and their mixture weights."""
        values = []
        weights = []
        for run in self.comparison.runs:
            result = run.result
            try:
                reconstruction = result.reconstructions[spectrum]
            except KeyError as error:
                raise ValueError(
                    f"model {result.name!r} has no intrinsic reconstruction "
                    f"for spectrum {spectrum!r}"
                ) from error
            values.append(reconstruction.integrated_intrinsic_flux)
            weights.append(
                self.model_probabilities[result.name] * _result_weights(result)
            )
        return np.concatenate(values), np.concatenate(weights)

    def integrated_flux_summary(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        """Summarize intrinsic flux while propagating model uncertainty."""
        values, weights = self.integrated_flux_mixture(spectrum)
        return summarize_integrated_flux(
            values,
            probabilities,
            weights=weights,
        )

    def as_dict(self):
        return {
            "assumption": "evidence-weighted Bayesian model average",
            "model_prior_probabilities": dict(self.model_prior_probabilities),
            "posterior_model_probabilities": dict(self.model_probabilities),
            "models": {
                run.result.name: {
                    "log_evidence": run.result.log_evidence,
                    "log_evidence_error": run.result.log_evidence_error,
                }
                for run in self.comparison.runs
            },
        }

    def plot(self, spectrum="spectrum", *, show_individual_models=True):
        return plot_model_average(
            self,
            spectrum=spectrum,
            show_individual_models=show_individual_models,
        )


def model_average(comparison, model_prior_probabilities=None):
    """Build an evidence-weighted mixture of completed model posteriors."""
    if not isinstance(comparison, InferenceComparisonResult):
        raise TypeError("comparison must be an InferenceComparisonResult")
    names = tuple(run.result.name for run in comparison.runs)
    if model_prior_probabilities is None:
        model_prior_probabilities = {name: 1.0 for name in names}
    priors = _normalized_probabilities(
        model_prior_probabilities,
        names,
        "model_prior_probabilities",
    )
    return ModelAveragedResult(
        comparison=comparison,
        model_prior_probabilities=priors,
        model_probabilities=posterior_model_probabilities(comparison, priors),
    )


def plot_model_average(
    result,
    spectrum="spectrum",
    *,
    show_individual_models=True,
):
    """Plot the data, model-average bands, and optional component medians."""
    import matplotlib.pyplot as plt

    if not isinstance(result, ModelAveragedResult):
        raise TypeError("result must be a ModelAveragedResult")
    first_result = result.comparison.runs[0].result
    try:
        observation = first_result.observations[spectrum]
    except KeyError as error:
        available = ", ".join(first_result.observations)
        raise KeyError(
            f"unknown spectrum {spectrum!r}; available spectra: {available}"
        ) from error
    bands = result.prediction_bands(spectrum)
    figure, axis = plt.subplots(figsize=(10.0, 5.5), layout="constrained")
    axis.errorbar(
        observation.wavelength,
        observation.flux,
        yerr=observation.error,
        color="0.55",
        linewidth=0.8,
        label="data (reported errors)",
    )
    axis.fill_between(
        observation.wavelength,
        bands[0],
        bands[4],
        color="tab:blue",
        alpha=0.14,
        label="95% model-averaged latent band",
    )
    axis.fill_between(
        observation.wavelength,
        bands[1],
        bands[3],
        color="tab:blue",
        alpha=0.24,
        label="68% model-averaged latent band",
    )
    axis.plot(
        observation.wavelength,
        bands[2],
        color="tab:blue",
        linewidth=2.2,
        label="model-averaged median",
    )
    if show_individual_models:
        colors = plt.get_cmap("tab10").colors
        for index, run in enumerate(result.comparison.runs):
            model_result = run.result
            median = posterior_percentile_bands(
                model_result.predictions[spectrum],
                probabilities=(0.5,),
                weights=model_result.weights,
            )[0]
            probability = result.model_probabilities[model_result.name]
            axis.plot(
                observation.wavelength,
                median,
                color=colors[(index + 1) % len(colors)],
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"{model_result.name} ({probability:.1%})",
            )
    axis.set_xlabel(r"wavelength ($\AA$)")
    axis.set_ylabel("flux density")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize="small", ncol=2)
    axis.set_title("Bayesian model-averaged latent detector prediction")
    return figure
