"""PSIS-LOO comparison and predictive stacking for completed NUTS fits."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.inference.workflow import InferenceResult, _json_compatible


def _arviz():
    try:
        import arviz as az
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PSIS-LOO requires ArviZ; install lyapy with "
            "`python -m pip install -e '.[comparison]'`"
        ) from error
    return az


def _spectrum_name(result, spectrum):
    if spectrum in result.observations:
        return spectrum
    available = ", ".join(repr(name) for name in result.observations)
    raise KeyError(
        f"unknown spectrum {spectrum!r}; available spectra: {available}"
    )


def _chain_shape(result):
    chains = int(result.diagnostics.get("chains", 1))
    draws_per_chain = int(
        result.diagnostics.get("draws_per_chain", result.draws // chains)
    )
    if chains < 1 or draws_per_chain < 1 or chains * draws_per_chain != result.draws:
        raise ValueError("result diagnostics contain an inconsistent chain shape")
    return chains, draws_per_chain


def _inference_data(result, spectrum):
    az = _arviz()
    spectrum = _spectrum_name(result, spectrum)
    if result.backend != "nuts":
        raise ValueError("PSIS-LOO currently requires an unweighted NUTS result")
    try:
        pointwise = result.pointwise_log_likelihoods[spectrum]
    except KeyError as error:
        raise ValueError(
            f"result {result.name!r} has no pointwise log likelihood for "
            f"spectrum {spectrum!r}"
        ) from error
    chains, draws_per_chain = _chain_shape(result)
    posterior = {
        name: result.samples[:, index].reshape(chains, draws_per_chain)
        for index, name in enumerate(result.parameter_names)
    }
    observation = result.observations[spectrum]
    likelihood_name = "observed_flux"
    return az.from_dict(
        posterior=posterior,
        log_likelihood={
            likelihood_name: pointwise.reshape(
                chains,
                draws_per_chain,
                observation.fitted_pixels,
            )
        },
        observed_data={
            likelihood_name: observation.flux[observation.likelihood_mask]
        },
        coords={"observation": observation.likelihood_indices},
        dims={likelihood_name: ["observation"]},
    )


@dataclass(frozen=True)
class PsisLooResult:
    """Pixel-level leave-one-out diagnostics for one NUTS fit."""

    model: str
    spectrum: str
    elpd_loo: float
    standard_error: float
    effective_parameters: float
    pointwise_elpd: np.ndarray
    pareto_k: np.ndarray
    wavelength: np.ndarray
    warning: bool
    good_k_threshold: float

    def __post_init__(self):
        if not self.model:
            raise ValueError("model must not be empty")
        pointwise = np.asarray(self.pointwise_elpd, dtype=np.float64)
        pareto_k = np.asarray(self.pareto_k, dtype=np.float64)
        wavelength = np.asarray(self.wavelength, dtype=np.float64)
        if pointwise.ndim != 1 or pointwise.size == 0:
            raise ValueError("pointwise_elpd must be a non-empty vector")
        if pareto_k.shape != pointwise.shape or wavelength.shape != pointwise.shape:
            raise ValueError(
                "pareto_k and wavelength must match pointwise_elpd"
            )
        scalar_values = (
            self.elpd_loo,
            self.standard_error,
            self.effective_parameters,
            self.good_k_threshold,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("PSIS-LOO scalar values must be finite")
        if self.standard_error < 0.0 or self.good_k_threshold <= 0.0:
            raise ValueError("PSIS-LOO uncertainty and threshold must be valid")
        if not np.all(np.isfinite(pointwise)) or np.any(np.isnan(pareto_k)):
            raise ValueError(
                "pointwise ELPD must be finite and Pareto k must not be NaN"
            )
        object.__setattr__(self, "pointwise_elpd", pointwise)
        object.__setattr__(self, "pareto_k", pareto_k)
        object.__setattr__(self, "wavelength", wavelength)

    @property
    def loo_information_criterion(self):
        return -2.0 * self.elpd_loo

    @property
    def maximum_pareto_k(self):
        return float(np.max(self.pareto_k))

    @property
    def pareto_k_above_threshold(self):
        return int(np.count_nonzero(self.pareto_k > self.good_k_threshold))

    @property
    def pareto_k_above_one(self):
        return int(np.count_nonzero(self.pareto_k > 1.0))

    @property
    def infinite_pareto_k(self):
        return int(np.count_nonzero(np.isinf(self.pareto_k)))

    def as_dict(self):
        maximum = self.maximum_pareto_k
        return {
            "model": self.model,
            "spectrum": self.spectrum,
            "elpd_loo": self.elpd_loo,
            "standard_error": self.standard_error,
            "loo_information_criterion": self.loo_information_criterion,
            "effective_parameters": self.effective_parameters,
            "warning": self.warning,
            "good_k_threshold": self.good_k_threshold,
            "maximum_pareto_k": (
                maximum if math.isfinite(maximum) else None
            ),
            "infinite_pareto_k": self.infinite_pareto_k,
            "pareto_k_above_threshold": self.pareto_k_above_threshold,
            "pareto_k_above_one": self.pareto_k_above_one,
            "wavelength": self.wavelength.tolist(),
            "pointwise_elpd": self.pointwise_elpd.tolist(),
            "pareto_k": [
                float(value) if math.isfinite(value) else None
                for value in self.pareto_k
            ],
        }

    def plot_pareto_k(self):
        """Plot influence diagnostics against fitted detector wavelength."""
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
        finite = np.isfinite(self.pareto_k)
        finite_values = self.pareto_k[finite]
        plot_ceiling = max(
            1.1,
            self.good_k_threshold * 1.1,
            float(np.max(finite_values)) * 1.1 if finite_values.size else 0.0,
        )
        axis.scatter(
            self.wavelength[finite],
            finite_values,
            color="tab:blue",
            s=20,
        )
        if np.any(~finite):
            axis.scatter(
                self.wavelength[~finite],
                np.full(np.count_nonzero(~finite), plot_ceiling),
                color="tab:red",
                marker="^",
                s=35,
                label=f"infinite k ({np.count_nonzero(~finite)} pixels)",
            )
        axis.axhline(0.5, color="0.6", linestyle=":", label="k = 0.5")
        axis.axhline(0.7, color="tab:orange", linestyle="--", label="k = 0.7")
        axis.axhline(1.0, color="tab:red", linestyle="--", label="k = 1")
        axis.axhline(
            self.good_k_threshold,
            color="black",
            linestyle="-.",
            label=f"sample-size threshold = {self.good_k_threshold:.2f}",
        )
        axis.set_xlabel(r"wavelength ($\AA$)")
        axis.set_ylabel("Pareto k")
        axis.set_ylim(top=plot_ceiling * 1.06)
        axis.set_title(f"PSIS-LOO influence diagnostics: {self.model}")
        axis.grid(alpha=0.2)
        axis.legend(
            frameon=False,
            fontsize="small",
            loc="center left",
        )
        return figure


def _psis_loo_result(result, spectrum, estimate):
    observation = result.observations[spectrum]
    return PsisLooResult(
        model=result.name,
        spectrum=spectrum,
        elpd_loo=float(estimate.elpd_loo),
        standard_error=float(estimate.se),
        effective_parameters=float(estimate.p_loo),
        pointwise_elpd=np.asarray(estimate.loo_i),
        pareto_k=np.asarray(estimate.pareto_k),
        wavelength=observation.wavelength[observation.likelihood_mask],
        warning=bool(estimate.warning),
        good_k_threshold=float(estimate.good_k),
    )


def evaluate_psis_loo(result, spectrum="spectrum"):
    """Evaluate pixel-level PSIS-LOO for one completed NUTS result."""
    if not isinstance(result, InferenceResult):
        raise TypeError("result must be an InferenceResult")
    spectrum = _spectrum_name(result, spectrum)
    estimate = _arviz().loo(
        _inference_data(result, spectrum),
        var_name="observed_flux",
        pointwise=True,
        scale="log",
    )
    return _psis_loo_result(result, spectrum, estimate)


@dataclass(frozen=True)
class PredictiveModelComparison:
    """One pairwise expected-log-predictive-density difference."""

    numerator: str
    denominator: str
    delta_elpd: float
    standard_error: float
    favored_model: str | None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PredictiveRanking:
    """One model ranked by expected out-of-sample predictive accuracy."""

    rank: int
    model: str
    elpd_loo: float
    standard_error: float
    elpd_difference_below_best: float
    difference_standard_error: float
    effective_parameters: float
    stacking_weight: float
    warning: bool
    maximum_pareto_k: float


def _same_observation(first, second):
    return all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in (
            "wavelength",
            "flux",
            "error",
            "likelihood_mask",
        )
    )


@dataclass(frozen=True)
class PsisLooComparisonResult:
    """PSIS-LOO rankings and stacking weights for matched NUTS fits."""

    results: tuple[InferenceResult, ...]
    loo_results: tuple[PsisLooResult, ...]
    stacking_weights: dict[str, float]
    spectrum: str

    def __post_init__(self):
        results = tuple(self.results)
        loo_results = tuple(self.loo_results)
        if len(results) < 2:
            raise ValueError("at least two NUTS results are required")
        if len(loo_results) != len(results):
            raise ValueError("loo_results must match results")
        if not all(isinstance(result, InferenceResult) for result in results):
            raise TypeError("results must contain InferenceResult values")
        if not all(result.backend == "nuts" for result in results):
            raise ValueError("PSIS-LOO comparison requires NUTS results")
        names = tuple(result.name for result in results)
        if len(names) != len(set(names)):
            raise ValueError("result names must be unique")
        if tuple(result.model for result in loo_results) != names:
            raise ValueError("loo_results must follow result order")
        weights = dict(self.stacking_weights)
        if set(weights) != set(names):
            raise ValueError("stacking_weights must match every model")
        values = np.asarray([weights[name] for name in names], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("stacking_weights must be finite and non-negative")
        if not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("stacking_weights must sum to one")
        reference = results[0].observations[self.spectrum]
        for result in results[1:]:
            try:
                observation = result.observations[self.spectrum]
            except KeyError as error:
                raise ValueError(
                    "all results must contain the selected spectrum"
                ) from error
            if not _same_observation(reference, observation):
                raise ValueError("all results must use identical observations")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "loo_results", loo_results)
        object.__setattr__(
            self,
            "stacking_weights",
            {name: float(weights[name]) for name in names},
        )

    @property
    def model_names(self):
        return tuple(result.name for result in self.results)

    @property
    def best_model(self):
        return max(self.loo_results, key=lambda result: result.elpd_loo).model

    @property
    def ranking(self):
        ordered = sorted(
            self.loo_results,
            key=lambda result: result.elpd_loo,
            reverse=True,
        )
        best = ordered[0]
        ranking = []
        for rank, result in enumerate(ordered, start=1):
            difference = best.pointwise_elpd - result.pointwise_elpd
            difference_standard_error = (
                0.0
                if result is best
                else float(
                    np.sqrt(
                        difference.size
                        * np.var(difference, ddof=1)
                    )
                )
            )
            ranking.append(
                PredictiveRanking(
                    rank=rank,
                    model=result.model,
                    elpd_loo=result.elpd_loo,
                    standard_error=result.standard_error,
                    elpd_difference_below_best=(
                        best.elpd_loo - result.elpd_loo
                    ),
                    difference_standard_error=difference_standard_error,
                    effective_parameters=result.effective_parameters,
                    stacking_weight=self.stacking_weights[result.model],
                    warning=result.warning,
                    maximum_pareto_k=result.maximum_pareto_k,
                )
            )
        return tuple(ranking)

    def get_result(self, model):
        for result in self.results:
            if result.name == model:
                return result
        raise KeyError(f"unknown model {model!r}")

    def get_loo_result(self, model):
        for result in self.loo_results:
            if result.model == model:
                return result
        raise KeyError(f"unknown model {model!r}")

    def compare(self, numerator, denominator):
        first = self.get_loo_result(numerator)
        second = self.get_loo_result(denominator)
        difference = first.pointwise_elpd - second.pointwise_elpd
        delta = float(np.sum(difference))
        standard_error = float(
            np.sqrt(difference.size * np.var(difference, ddof=1))
        )
        if delta > 0.0:
            favored = numerator
        elif delta < 0.0:
            favored = denominator
        else:
            favored = None
        return PredictiveModelComparison(
            numerator=numerator,
            denominator=denominator,
            delta_elpd=delta,
            standard_error=standard_error,
            favored_model=favored,
        )

    def prediction_mixture(self):
        samples = []
        weights = []
        for result in self.results:
            prediction = result.predictions[self.spectrum]
            samples.append(prediction)
            weights.append(
                np.full(
                    result.draws,
                    self.stacking_weights[result.name] / result.draws,
                )
            )
        return np.concatenate(samples), np.concatenate(weights)

    def prediction_bands(
        self,
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        samples, weights = self.prediction_mixture()
        return posterior_percentile_bands(
            samples,
            probabilities=probabilities,
            weights=weights,
        )

    def integrated_flux_summary(
        self,
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        values = []
        weights = []
        for result in self.results:
            try:
                flux = result.reconstructions[
                    self.spectrum
                ].integrated_intrinsic_flux
            except KeyError as error:
                raise ValueError(
                    f"model {result.name!r} has no intrinsic reconstruction "
                    f"for spectrum {self.spectrum!r}"
                ) from error
            values.append(flux)
            weights.append(
                np.full(
                    result.draws,
                    self.stacking_weights[result.name] / result.draws,
                )
            )
        return summarize_integrated_flux(
            np.concatenate(values),
            probabilities=probabilities,
            weights=np.concatenate(weights),
        )

    def as_dict(self):
        ranking = []
        for item in self.ranking:
            values = asdict(item)
            if not math.isfinite(values["maximum_pareto_k"]):
                values["maximum_pareto_k"] = None
            values["infinite_pareto_k"] = self.get_loo_result(
                item.model
            ).infinite_pareto_k
            ranking.append(values)
        return {
            "method": "pixel-level PSIS-LOO with predictive stacking",
            "spectrum": self.spectrum,
            "best_predictive_model": self.best_model,
            "ranking": ranking,
            "stacking_weights": dict(self.stacking_weights),
            "models": {
                result.model: result.as_dict()
                for result in self.loo_results
            },
            "caveat": (
                "Pixel-level holdout can be optimistic for oversampled or "
                "correlated spectra; inspect Pareto k and use blocked "
                "cross-validation for a stronger test."
            ),
        }

    def save(self, path):
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_compatible(self.as_dict()), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def plot(self):
        """Plot ELPD estimates and predictive stacking weights."""
        import matplotlib.pyplot as plt

        ranking = self.ranking
        labels = [item.model for item in ranking]
        positions = np.arange(len(ranking))
        figure, (elpd_axis, weight_axis) = plt.subplots(
            2,
            1,
            figsize=(9.0, 7.0),
            layout="constrained",
        )
        elpd_axis.errorbar(
            positions,
            [item.elpd_loo for item in ranking],
            yerr=[item.standard_error for item in ranking],
            marker="o",
            linestyle="none",
            capsize=4,
            color="tab:blue",
        )
        elpd_axis.set_xticks(positions, labels, rotation=20, ha="right")
        elpd_axis.set_ylabel("ELPD-LOO (higher is better)")
        elpd_axis.grid(axis="y", alpha=0.2)
        weight_axis.bar(
            positions,
            [item.stacking_weight for item in ranking],
            color="tab:orange",
        )
        weight_axis.set_xticks(positions, labels, rotation=20, ha="right")
        weight_axis.set_ylabel("predictive stacking weight")
        weight_axis.set_ylim(0.0, 1.0)
        weight_axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"Pixel-level PSIS-LOO: {self.spectrum}")
        return figure

    def plot_average(self, *, show_individual_models=True):
        """Plot the stacking-weighted latent detector prediction."""
        import matplotlib.pyplot as plt

        observation = self.results[0].observations[self.spectrum]
        bands = self.prediction_bands()
        figure, axis = plt.subplots(figsize=(10.0, 5.5), layout="constrained")
        included = observation.likelihood_mask
        axis.errorbar(
            observation.wavelength[included],
            observation.flux[included],
            yerr=observation.error[included],
            color="0.55",
            linewidth=0.8,
            label="data (reported errors)",
        )
        if np.any(~included):
            axis.errorbar(
                observation.wavelength[~included],
                observation.flux[~included],
                yerr=observation.error[~included],
                color="tab:red",
                marker="x",
                linestyle="none",
                linewidth=0.8,
                label="excluded from likelihood",
            )
        axis.fill_between(
            observation.wavelength,
            bands[0],
            bands[4],
            color="tab:blue",
            alpha=0.14,
            label="95% stacking-weighted latent band",
        )
        axis.fill_between(
            observation.wavelength,
            bands[1],
            bands[3],
            color="tab:blue",
            alpha=0.24,
            label="68% stacking-weighted latent band",
        )
        axis.plot(
            observation.wavelength,
            bands[2],
            color="tab:blue",
            linewidth=2.2,
            label="stacking-weighted median",
        )
        if show_individual_models:
            colors = plt.get_cmap("tab10").colors
            for index, result in enumerate(self.results):
                median = result.prediction_bands(
                    self.spectrum,
                    probabilities=(0.5,),
                )[0]
                axis.plot(
                    observation.wavelength,
                    median,
                    color=colors[(index + 1) % len(colors)],
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                    label=(
                        f"{result.name} "
                        f"({self.stacking_weights[result.name]:.1%})"
                    ),
                )
        axis.set_xlabel(r"wavelength ($\AA$)")
        axis.set_ylabel("flux density")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize="small", ncol=2)
        axis.set_title("PSIS-LOO predictive stacking")
        return figure


def compare_nuts_results(results, spectrum="spectrum"):
    """Compare matched NUTS fits with PSIS-LOO and predictive stacking."""
    results = tuple(results)
    if len(results) < 2:
        raise ValueError("at least two NUTS results are required")
    if not all(isinstance(result, InferenceResult) for result in results):
        raise TypeError("results must contain InferenceResult values")
    names = [result.name for result in results]
    if len(names) != len(set(names)):
        raise ValueError("result names must be unique")
    spectrum = _spectrum_name(results[0], spectrum)
    reference = results[0].observations[spectrum]
    for result in results[1:]:
        current_spectrum = _spectrum_name(result, spectrum)
        if not _same_observation(
            reference,
            result.observations[current_spectrum],
        ):
            raise ValueError("all results must use identical observations")
    inference_data = {
        result.name: _inference_data(result, spectrum)
        for result in results
    }
    az = _arviz()
    estimates = {
        name: az.loo(
            data,
            var_name="observed_flux",
            pointwise=True,
            scale="log",
        )
        for name, data in inference_data.items()
    }
    loo_results = tuple(
        _psis_loo_result(result, spectrum, estimates[result.name])
        for result in results
    )
    comparison = az.compare(
        estimates,
        ic="loo",
        method="stacking",
        scale="log",
    )
    stacking_weights = {
        name: max(0.0, float(comparison.loc[name, "weight"]))
        for name in names
    }
    total = sum(stacking_weights.values())
    stacking_weights = {
        name: weight / total
        for name, weight in stacking_weights.items()
    }
    return PsisLooComparisonResult(
        results=results,
        loo_results=loo_results,
        stacking_weights=stacking_weights,
        spectrum=spectrum,
    )
