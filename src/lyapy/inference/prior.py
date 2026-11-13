"""Prior-predictive checks, boundary diagnostics, and sensitivity runs."""

from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from lyapy.inference.common import InferenceConfiguration, SpectrumData
from lyapy.inference.diagnostics import summarize_parameters, weighted_quantile
from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    ReconstructionSamples,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.inference.workflow import (
    InferenceComparisonResult,
    InferenceProblem,
    _evaluate_predictions,
    _evaluate_reconstructions,
    _json_compatible,
    _observation_mapping,
    _same_observations,
    compare_inference_problems,
    load_inference_comparison,
)


PRIOR_SENSITIVITY_SCHEMA_VERSION = 1


def with_prior_bounds(configuration, bounds):
    """Return a configuration with explicit replacement prior bounds."""
    if not isinstance(configuration, InferenceConfiguration):
        raise TypeError("configuration must be an InferenceConfiguration")
    bounds = dict(bounds)
    known = {parameter.name for parameter in configuration.free_parameters}
    unknown = set(bounds) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise KeyError(f"unknown free parameter(s): {names}")
    free_parameters = []
    for parameter in configuration.free_parameters:
        if parameter.name not in bounds:
            free_parameters.append(parameter)
            continue
        lower, upper = bounds[parameter.name]
        replacement = replace(
            parameter,
            lower=float(lower),
            upper=float(upper),
        )
        initial = configuration.base_parameters[parameter.index]
        if not replacement.lower <= initial <= replacement.upper:
            raise ValueError(
                f"initial value for {parameter.name!r} lies outside new bounds"
            )
        free_parameters.append(replacement)
    return InferenceConfiguration(
        base_parameters=configuration.base_parameters,
        free_parameters=tuple(free_parameters),
        parameter_names=configuration.parameter_names,
    )


def sample_prior(configuration, draws=1000, random_seed=0):
    """Draw free-parameter vectors from a configuration's declared priors."""
    if not isinstance(configuration, InferenceConfiguration):
        raise TypeError("configuration must be an InferenceConfiguration")
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    random = np.random.default_rng(random_seed)
    unit_values = random.random((draws, configuration.ndim))
    return np.vstack(
        [configuration.transform_unit_cube(values) for values in unit_values]
    )


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Parameter draws and spectra implied by one problem's priors."""

    name: str
    random_seed: int
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    problem_specification: dict
    observations: dict[str, SpectrumData]
    predictions: dict[str, np.ndarray]
    reconstructions: dict[str, ReconstructionSamples] = field(default_factory=dict)

    def __post_init__(self):
        samples = np.asarray(self.samples, dtype=np.float64)
        parameter_names = tuple(self.parameter_names)
        observations = dict(self.observations)
        predictions = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in self.predictions.items()
        }
        reconstructions = dict(self.reconstructions)
        if not self.name:
            raise ValueError("prior-predictive name must not be empty")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError("samples must have shape (draws, parameters)")
        if samples.shape[1] != len(parameter_names):
            raise ValueError("samples must match parameter_names")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("parameter_names must be unique")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must contain finite values")
        if not all(
            isinstance(observation, SpectrumData)
            for observation in observations.values()
        ):
            raise TypeError("observations must contain SpectrumData values")
        if not observations or set(predictions) != set(observations):
            raise ValueError("predictions must match all named observations")
        for spectrum, prediction in predictions.items():
            expected = (samples.shape[0], observations[spectrum].wavelength.size)
            if prediction.shape != expected:
                raise ValueError(
                    f"prediction {spectrum!r} must have shape {expected}"
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError("predictions must contain finite values")
        if not set(reconstructions).issubset(observations):
            raise ValueError("reconstructions must match named observations")
        if not all(
            isinstance(value, ReconstructionSamples)
            for value in reconstructions.values()
        ):
            raise TypeError(
                "reconstructions must contain ReconstructionSamples values"
            )
        if not isinstance(self.problem_specification, dict):
            raise TypeError("problem_specification must be a dictionary")
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "reconstructions", reconstructions)

    @property
    def draws(self):
        return self.samples.shape[0]

    def parameter_summary(self, interval_probability=0.9):
        values = {
            name: self.samples[:, index]
            for index, name in enumerate(self.parameter_names)
        }
        return summarize_parameters(
            values,
            interval_probability=interval_probability,
        )

    def prediction_bands(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        try:
            predictions = self.predictions[spectrum]
        except KeyError as error:
            raise KeyError(f"unknown spectrum {spectrum!r}") from error
        return posterior_percentile_bands(predictions, probabilities)

    def integrated_flux_summary(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        try:
            values = self.reconstructions[
                spectrum
            ].integrated_intrinsic_flux
        except KeyError as error:
            raise ValueError(
                f"spectrum {spectrum!r} has no intrinsic reconstruction"
            ) from error
        return summarize_integrated_flux(values, probabilities)

    def as_dict(self):
        return {
            "name": self.name,
            "random_seed": self.random_seed,
            "draws": self.draws,
            "parameter_names": list(self.parameter_names),
            "problem": self.problem_specification,
            "spectra": {
                name: {
                    "pixels": int(observation.wavelength.size),
                    "has_intrinsic_reconstruction": name in self.reconstructions,
                }
                for name, observation in self.observations.items()
            },
        }

    def plot(self, spectrum="spectrum", *, maximum_curves=30):
        return plot_prior_predictive(
            self,
            spectrum=spectrum,
            maximum_curves=maximum_curves,
        )


def sample_prior_predictive(
    problem,
    *,
    draws=1000,
    random_seed=0,
    reconstruct=True,
    reconstruction_batch_size=None,
):
    """Evaluate detector and optional intrinsic spectra drawn from the priors."""
    if not isinstance(problem, InferenceProblem):
        raise TypeError("problem must be an InferenceProblem")
    if reconstruction_batch_size is not None and (
        not isinstance(reconstruction_batch_size, int)
        or reconstruction_batch_size < 1
    ):
        raise ValueError(
            "reconstruction_batch_size must be a positive integer or None"
        )
    samples = sample_prior(
        problem.configuration,
        draws=draws,
        random_seed=random_seed,
    )
    observations = _observation_mapping(problem)
    predictions = _evaluate_predictions(problem, samples)
    reconstructions = (
        _evaluate_reconstructions(
            problem,
            samples,
            reconstruction_batch_size,
        )
        if reconstruct
        else {}
    )
    return PriorPredictiveResult(
        name=problem.name,
        random_seed=random_seed,
        parameter_names=tuple(
            parameter.name for parameter in problem.configuration.free_parameters
        ),
        samples=samples,
        problem_specification=problem.as_dict(),
        observations=observations,
        predictions=predictions,
        reconstructions=reconstructions,
    )


def plot_prior_predictive(result, spectrum="spectrum", *, maximum_curves=30):
    """Plot observed data against spectra implied by the declared priors."""
    import matplotlib.pyplot as plt

    if not isinstance(maximum_curves, int) or maximum_curves < 0:
        raise ValueError("maximum_curves must be a non-negative integer")
    try:
        observation = result.observations[spectrum]
        predictions = result.predictions[spectrum]
    except KeyError as error:
        raise KeyError(f"unknown spectrum {spectrum!r}") from error
    reconstruction = result.reconstructions.get(spectrum)
    rows = 2 if reconstruction is not None else 1
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=(9.5, 8.0 if rows == 2 else 5.0),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    detector_axis = axes[0]
    bands = result.prediction_bands(spectrum)
    detector_axis.errorbar(
        observation.wavelength,
        observation.flux,
        yerr=observation.error,
        color="black",
        linewidth=0.7,
        alpha=0.65,
        label="observed data",
    )
    if maximum_curves:
        indices = np.linspace(
            0,
            result.draws - 1,
            min(maximum_curves, result.draws),
            dtype=int,
        )
        detector_axis.plot(
            observation.wavelength,
            predictions[indices].T,
            color="tab:blue",
            alpha=0.08,
            linewidth=0.8,
        )
    detector_axis.fill_between(
        observation.wavelength,
        bands[0],
        bands[4],
        color="tab:blue",
        alpha=0.15,
        label="95% prior-predictive latent band",
    )
    detector_axis.plot(
        observation.wavelength,
        bands[2],
        color="tab:blue",
        label="prior-predictive median",
    )
    detector_axis.set_ylabel("flux density")
    detector_axis.set_xlabel(r"wavelength ($\AA$)")
    detector_axis.legend(frameon=False, fontsize="small")
    if reconstruction is not None:
        intrinsic_axis = axes[1]
        intrinsic_bands = posterior_percentile_bands(
            reconstruction.intrinsic
        )
        intrinsic_axis.fill_between(
            reconstruction.model_wavelength,
            intrinsic_bands[0],
            intrinsic_bands[4],
            color="tab:orange",
            alpha=0.2,
            label="95% intrinsic prior band",
        )
        intrinsic_axis.plot(
            reconstruction.model_wavelength,
            intrinsic_bands[2],
            color="tab:orange",
            label="intrinsic prior median",
        )
        intrinsic_axis.set_ylabel("intrinsic flux density")
        intrinsic_axis.set_xlabel(r"wavelength ($\AA$)")
        intrinsic_axis.legend(frameon=False, fontsize="small")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(f"Prior-predictive check: {result.name}")
    return figure


@dataclass(frozen=True)
class PriorBoundaryDiagnostic:
    """Posterior mass near one bounded prior's lower and upper edges."""

    parameter: str
    lower: float
    upper: float
    posterior_median: float
    lower_boundary_probability: float
    upper_boundary_probability: float
    boundary_fraction: float
    warning_probability: float
    boundary_distance_in_posterior_widths: float
    lower_distance_in_posterior_widths: float | None
    upper_distance_in_posterior_widths: float | None
    pressure: str

    def as_dict(self):
        return asdict(self)


def analyze_prior_boundaries(
    result,
    *,
    boundary_fraction=0.05,
    warning_probability=0.2,
    boundary_distance_in_posterior_widths=0.5,
):
    """Report posterior mass in fixed-width regions at each prior boundary."""
    if not 0.0 < boundary_fraction < 0.5:
        raise ValueError("boundary_fraction must lie between zero and 0.5")
    if not 0.0 < warning_probability <= 1.0:
        raise ValueError("warning_probability must lie between zero and one")
    if boundary_distance_in_posterior_widths < 0.0:
        raise ValueError(
            "boundary_distance_in_posterior_widths must be non-negative"
        )
    specifications = {
        parameter["name"]: parameter
        for parameter in result.problem_specification["hypothesis"][
            "free_parameters"
        ]
    }
    diagnostics = []
    normalized_weights = (
        np.full(result.draws, 1.0 / result.draws)
        if result.weights is None
        else result.weights / np.sum(result.weights)
    )
    for column, name in enumerate(result.parameter_names):
        specification = specifications[name]
        lower = float(specification["lower"])
        upper = float(specification["upper"])
        width = upper - lower
        values = result.samples[:, column]
        lower_probability = float(
            np.sum(normalized_weights[values <= lower + boundary_fraction * width])
        )
        upper_probability = float(
            np.sum(normalized_weights[values >= upper - boundary_fraction * width])
        )
        posterior_lower, posterior_upper = weighted_quantile(
            values,
            (0.05, 0.95),
            result.weights,
        )
        posterior_width = posterior_upper - posterior_lower
        if posterior_width == 0.0:
            lower_distance = 0.0 if posterior_lower == lower else None
            upper_distance = 0.0 if posterior_upper == upper else None
        else:
            lower_distance = (posterior_lower - lower) / posterior_width
            upper_distance = (upper - posterior_upper) / posterior_width
        lower_pressure = (
            lower_probability >= warning_probability
            and lower_distance is not None
            and lower_distance <= boundary_distance_in_posterior_widths
        )
        upper_pressure = (
            upper_probability >= warning_probability
            and upper_distance is not None
            and upper_distance <= boundary_distance_in_posterior_widths
        )
        if lower_pressure and upper_pressure:
            pressure = "both"
        elif lower_pressure:
            pressure = "lower"
        elif upper_pressure:
            pressure = "upper"
        else:
            pressure = "none"
        diagnostics.append(
            PriorBoundaryDiagnostic(
                parameter=name,
                lower=lower,
                upper=upper,
                posterior_median=float(
                    weighted_quantile(values, 0.5, result.weights)
                ),
                lower_boundary_probability=lower_probability,
                upper_boundary_probability=upper_probability,
                boundary_fraction=float(boundary_fraction),
                warning_probability=float(warning_probability),
                boundary_distance_in_posterior_widths=float(
                    boundary_distance_in_posterior_widths
                ),
                lower_distance_in_posterior_widths=(
                    None if lower_distance is None else float(lower_distance)
                ),
                upper_distance_in_posterior_widths=(
                    None if upper_distance is None else float(upper_distance)
                ),
                pressure=pressure,
            )
        )
    return tuple(diagnostics)


@dataclass(frozen=True)
class PriorSensitivityScenario:
    """One named set of matched model problems with explicit priors."""

    name: str
    problems: tuple[InferenceProblem, ...]
    description: str = ""

    def __post_init__(self):
        problems = tuple(self.problems)
        if not self.name:
            raise ValueError("scenario name must not be empty")
        if len(problems) < 2:
            raise ValueError("a sensitivity scenario requires at least two problems")
        if not all(isinstance(problem, InferenceProblem) for problem in problems):
            raise TypeError("scenario problems must be InferenceProblem values")
        names = [problem.name for problem in problems]
        if len(names) != len(set(names)):
            raise ValueError("scenario problem names must be unique")
        object.__setattr__(self, "problems", problems)


@dataclass(frozen=True)
class PriorSensitivityScenarioResult:
    """One named scenario and its completed model comparison."""

    name: str
    description: str
    comparison: InferenceComparisonResult

    def __post_init__(self):
        if not self.name:
            raise ValueError("scenario-result name must not be empty")
        if not isinstance(self.comparison, InferenceComparisonResult):
            raise TypeError("comparison must be an InferenceComparisonResult")


@dataclass(frozen=True)
class PriorSensitivityResult:
    """Evidence rankings across several explicit prior scenarios."""

    scenarios: tuple[PriorSensitivityScenarioResult, ...]

    def __post_init__(self):
        scenarios = tuple(self.scenarios)
        if len(scenarios) < 2:
            raise ValueError("at least two prior scenarios are required")
        if not all(
            isinstance(scenario, PriorSensitivityScenarioResult)
            for scenario in scenarios
        ):
            raise TypeError(
                "scenarios must contain PriorSensitivityScenarioResult values"
            )
        names = [scenario.name for scenario in scenarios]
        if len(names) != len(set(names)):
            raise ValueError("prior scenario names must be unique")
        model_names = tuple(
            run.result.name for run in scenarios[0].comparison.runs
        )
        for scenario in scenarios[1:]:
            current = tuple(run.result.name for run in scenario.comparison.runs)
            if current != model_names:
                raise ValueError("all scenarios must contain models in the same order")
        object.__setattr__(self, "scenarios", scenarios)

    @property
    def model_names(self):
        return tuple(
            run.result.name for run in self.scenarios[0].comparison.runs
        )

    @property
    def winner_stable(self):
        return len(
            {scenario.comparison.best_model for scenario in self.scenarios}
        ) == 1

    @property
    def ranking_stable(self):
        rankings = {
            tuple(item.model for item in scenario.comparison.ranking)
            for scenario in self.scenarios
        }
        return len(rankings) == 1

    def pairwise_summary(self, numerator, denominator):
        """Summarize one evidence contrast across all prior scenarios."""
        records = []
        for scenario in self.scenarios:
            comparison = scenario.comparison.compare(numerator, denominator)
            records.append(
                {
                    "scenario": scenario.name,
                    "delta_log_evidence": comparison.delta_log_evidence,
                    "delta_log_evidence_error": (
                        comparison.delta_log_evidence_error
                    ),
                    "favored_model": comparison.favored_model,
                    "strength": comparison.strength,
                }
            )
        minimum = min(records, key=lambda record: record["delta_log_evidence"])
        maximum = max(records, key=lambda record: record["delta_log_evidence"])
        span = maximum["delta_log_evidence"] - minimum["delta_log_evidence"]
        span_error = math.hypot(
            maximum["delta_log_evidence_error"],
            minimum["delta_log_evidence_error"],
        )
        return {
            "numerator": numerator,
            "denominator": denominator,
            "favored_model_stable": len(
                {record["favored_model"] for record in records}
            )
            == 1,
            "strength_stable": len({record["strength"] for record in records})
            == 1,
            "minimum_delta_log_evidence": minimum["delta_log_evidence"],
            "maximum_delta_log_evidence": maximum["delta_log_evidence"],
            "delta_log_evidence_span": span,
            "span_error": span_error,
            "span_relative_to_error": (
                None if span_error == 0.0 else span / span_error
            ),
            "scenarios": records,
        }

    def as_dict(self):
        pairwise = {
            f"{first} vs {second}": self.pairwise_summary(first, second)
            for first, second in combinations(self.model_names, 2)
        }
        return {
            "schema_version": PRIOR_SENSITIVITY_SCHEMA_VERSION,
            "winner_stable": self.winner_stable,
            "ranking_stable": self.ranking_stable,
            "model_names": list(self.model_names),
            "scenarios": {
                scenario.name: {
                    "description": scenario.description,
                    "comparison": scenario.comparison.as_dict(),
                }
                for scenario in self.scenarios
            },
            "pairwise_sensitivity": pairwise,
        }

    def save(self, directory):
        """Save every comparison and one cross-scenario sensitivity manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.as_dict()
        for index, scenario in enumerate(self.scenarios):
            manifest, _ = scenario.comparison.save(
                directory / f"scenario_{index}"
            )
            metadata["scenarios"][scenario.name]["comparison_file"] = str(
                manifest.relative_to(directory)
            )
        path = directory / "prior_sensitivity.json"
        path.write_text(
            json.dumps(_json_compatible(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def plot(self, numerator, denominator):
        return plot_prior_sensitivity(self, numerator, denominator)


def _same_component(first, second):
    if first is second:
        return True
    first_specification = getattr(first, "as_dict", None)
    second_specification = getattr(second, "as_dict", None)
    if callable(first_specification) and callable(second_specification):
        return first_specification() == second_specification()
    return False


def _same_model_except_priors(first, second):
    first_configuration = first.configuration
    second_configuration = second.configuration
    first_free = tuple(
        (parameter.name, parameter.index)
        for parameter in first_configuration.free_parameters
    )
    second_free = tuple(
        (parameter.name, parameter.index)
        for parameter in second_configuration.free_parameters
    )
    return (
        first.name == second.name
        and first_configuration.parameter_names
        == second_configuration.parameter_names
        and np.array_equal(
            first_configuration.base_parameters,
            second_configuration.base_parameters,
        )
        and first_free == second_free
        and _same_component(first.forward_model, second.forward_model)
        and _same_component(
            first.likelihood_function,
            second.likelihood_function,
        )
    )


def run_prior_sensitivity(
    scenarios,
    *,
    random_seed=0,
    progress=True,
    reconstruct=True,
    reconstruction_batch_size=None,
    **dynesty_options,
):
    """Run matched model comparisons under two or more named prior scenarios."""
    scenarios = tuple(scenarios)
    if len(scenarios) < 2:
        raise ValueError("at least two prior scenarios are required")
    if not all(isinstance(item, PriorSensitivityScenario) for item in scenarios):
        raise TypeError("scenarios must contain PriorSensitivityScenario values")
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("prior scenario names must be unique")
    model_names = tuple(problem.name for problem in scenarios[0].problems)
    reference = scenarios[0].problems[0]
    reference_models = scenarios[0].problems
    for scenario in scenarios:
        if tuple(problem.name for problem in scenario.problems) != model_names:
            raise ValueError("all scenarios must contain models in the same order")
        if not all(
            _same_observations(reference, problem)
            for problem in scenario.problems
        ):
            raise ValueError("all scenarios must use identical observations")
        if not all(
            _same_model_except_priors(reference_problem, problem)
            for reference_problem, problem in zip(
                reference_models,
                scenario.problems,
            )
        ):
            raise ValueError(
                "prior scenarios may change priors but not model structure"
            )
    results = []
    seed_offset = 0
    for scenario in scenarios:
        comparison = compare_inference_problems(
            scenario.problems,
            random_seed=random_seed + seed_offset,
            progress=progress,
            reconstruct=reconstruct,
            reconstruction_batch_size=reconstruction_batch_size,
            **dynesty_options,
        )
        results.append(
            PriorSensitivityScenarioResult(
                scenario.name,
                scenario.description,
                comparison,
            )
        )
        seed_offset += len(scenario.problems)
    return PriorSensitivityResult(tuple(results))


def load_prior_sensitivity(path):
    """Load a sensitivity collection written by PriorSensitivityResult.save."""
    manifest = Path(path)
    if manifest.suffix != ".json":
        manifest = manifest / "prior_sensitivity.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != PRIOR_SENSITIVITY_SCHEMA_VERSION:
        raise ValueError("unsupported prior-sensitivity schema version")
    scenarios = []
    for name, scenario in metadata["scenarios"].items():
        scenarios.append(
            PriorSensitivityScenarioResult(
                name=name,
                description=scenario["description"],
                comparison=load_inference_comparison(
                    manifest.parent / scenario["comparison_file"]
                ),
            )
        )
    result = PriorSensitivityResult(tuple(scenarios))
    if (
        result.winner_stable != metadata["winner_stable"]
        or result.ranking_stable != metadata["ranking_stable"]
    ):
        raise ValueError("prior-sensitivity manifest is inconsistent")
    return result


def plot_prior_sensitivity(result, numerator, denominator):
    """Plot one pairwise evidence contrast across named prior scenarios."""
    import matplotlib.pyplot as plt

    summary = result.pairwise_summary(numerator, denominator)
    records = summary["scenarios"]
    values = np.asarray(
        [record["delta_log_evidence"] for record in records]
    )
    errors = np.asarray(
        [record["delta_log_evidence_error"] for record in records]
    )
    labels = [record["scenario"] for record in records]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    positions = np.arange(len(records))
    axis.axhline(0.0, color="0.4", linewidth=1.0)
    axis.errorbar(
        positions,
        values,
        yerr=errors,
        color="tab:blue",
        marker="o",
        capsize=4,
        linestyle="none",
    )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel(r"$\Delta \log Z$ (numerator - denominator)")
    axis.set_title(
        f"Prior sensitivity: {numerator} vs {denominator}; "
        f"favored model stable = {summary['favored_model_stable']}"
    )
    axis.grid(axis="y", alpha=0.2)
    return figure
