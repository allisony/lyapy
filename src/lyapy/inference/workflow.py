"""High-level inference problems, backend-neutral results, and persistence."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from lyapy.inference.common import InferenceConfiguration, SpectrumData
from lyapy.inference.comparison import (
    EvidenceEstimate,
    ModelHypothesis,
    compare_evidence,
    rank_evidence,
)
from lyapy.inference.diagnostics import (
    summarize_dynesty,
    summarize_nuts,
    summarize_parameters,
)
from lyapy.inference.joint import (
    JointForwardModel,
    JointLikelihood,
    JointSpectrumData,
)
from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    ReconstructionEvaluator,
    ReconstructionSamples,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.likelihoods import gaussian_log_likelihood


RESULT_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1


def _spectrum_specification(data):
    return {
        "pixels": int(data.wavelength.size),
        "wavelength_minimum": float(np.min(data.wavelength)),
        "wavelength_maximum": float(np.max(data.wavelength)),
    }


@dataclass(frozen=True)
class InferenceProblem:
    """Data, parameters, forward model, and likelihood for one fit."""

    data: SpectrumData | JointSpectrumData
    configuration: InferenceConfiguration
    forward_model: object
    likelihood_function: object = gaussian_log_likelihood
    name: str = "spectrum inference"

    def __post_init__(self):
        if not self.name:
            raise ValueError("problem name must not be empty")
        if not isinstance(self.data, (SpectrumData, JointSpectrumData)):
            raise TypeError("data must be SpectrumData or JointSpectrumData")
        if not isinstance(self.configuration, InferenceConfiguration):
            raise TypeError("configuration must be an InferenceConfiguration")
        if not callable(self.forward_model):
            raise TypeError("forward_model must be callable")
        if not callable(self.likelihood_function):
            raise TypeError("likelihood_function must be callable")
        if isinstance(self.data, JointSpectrumData):
            if not callable(getattr(self.forward_model, "predict", None)):
                raise TypeError("a joint forward_model must define predict")
            if (
                isinstance(self.forward_model, JointForwardModel)
                and set(self.data.names) != set(self.forward_model.names)
            ):
                raise ValueError("joint data and forward-model names must match")
            if (
                isinstance(self.likelihood_function, JointLikelihood)
                and set(self.data.names)
                != {
                    likelihood.name
                    for likelihood in self.likelihood_function.likelihoods
                }
            ):
                raise ValueError("joint data and likelihood names must match")

    def hypothesis(self):
        """Return the corresponding low-level model hypothesis."""
        return ModelHypothesis(
            self.name,
            self.configuration,
            forward_model=self.forward_model,
            likelihood_function=self.likelihood_function,
        )

    def as_dict(self):
        """Return a JSON-compatible reproducibility specification."""
        if isinstance(self.data, JointSpectrumData):
            data_specification = {
                "type": "joint",
                "spectra": {
                    spectrum.name: _spectrum_specification(spectrum.data)
                    for spectrum in self.data.spectra
                },
            }
        else:
            data_specification = {
                "type": "single",
                "spectra": {"spectrum": _spectrum_specification(self.data)},
            }
        return {
            "name": self.name,
            "data": data_specification,
            "hypothesis": self.hypothesis().as_dict(),
        }

    def run(
        self,
        backend,
        *,
        random_seed=0,
        progress=True,
        reconstruct=True,
        reconstruction_batch_size=None,
        **backend_options,
    ):
        """Run NUTS or dynesty through the common workflow interface."""
        return run_inference(
            self,
            backend,
            random_seed=random_seed,
            progress=progress,
            reconstruct=reconstruct,
            reconstruction_batch_size=reconstruction_batch_size,
            **backend_options,
        )

    def compare(
        self,
        *alternatives,
        random_seed=0,
        progress=True,
        reconstruct=True,
        reconstruction_batch_size=None,
        **dynesty_options,
    ):
        """Compare this problem with alternatives using Bayesian evidence."""
        return compare_inference_problems(
            (self, *alternatives),
            random_seed=random_seed,
            progress=progress,
            reconstruct=reconstruct,
            reconstruction_batch_size=reconstruction_batch_size,
            **dynesty_options,
        )

    def prior_predictive(
        self,
        *,
        draws=1000,
        random_seed=0,
        reconstruct=True,
        reconstruction_batch_size=None,
    ):
        """Draw parameters and spectra directly from the declared priors."""
        from lyapy.inference.prior import sample_prior_predictive

        return sample_prior_predictive(
            self,
            draws=draws,
            random_seed=random_seed,
            reconstruct=reconstruct,
            reconstruction_batch_size=reconstruction_batch_size,
        )


def _observation_mapping(problem):
    if isinstance(problem.data, JointSpectrumData):
        return {
            spectrum.name: spectrum.data
            for spectrum in problem.data.spectra
        }
    return {"spectrum": problem.data}


def _evaluate_predictions(problem, free_values):
    values = jnp.asarray(free_values)
    if isinstance(problem.data, JointSpectrumData):

        def evaluate_one(draw):
            parameters = problem.configuration.assemble_jax(draw)
            return problem.forward_model.predict(problem.data, parameters)

        predictions = tuple(jax.jit(jax.vmap(evaluate_one))(values))
        if len(predictions) != len(problem.data.names):
            raise ValueError(
                "joint forward-model predictions must match the spectra"
            )
        return {
            name: np.asarray(prediction)
            for name, prediction in zip(problem.data.names, predictions)
        }

    wave = jnp.asarray(problem.data.wavelength)
    kernel = jnp.asarray(problem.data.convolution_kernel)

    def evaluate_one(draw):
        parameters = problem.configuration.assemble_jax(draw)
        return problem.forward_model(
            wave,
            parameters,
            kernel,
        ).convolved.observed

    prediction = jax.jit(jax.vmap(evaluate_one))(values)
    return {"spectrum": np.asarray(prediction)}


def _supports_physical_reconstruction(data, configuration, forward_model):
    result = forward_model(
        data.wavelength,
        jnp.asarray(configuration.base_parameters),
        data.convolution_kernel,
    )
    return (
        hasattr(result, "unconvolved")
        and hasattr(result.unconvolved, "intrinsic")
        and hasattr(result, "convolved")
        and hasattr(result.convolved, "observed")
    )


def _evaluate_reconstructions(problem, free_values, batch_size):
    if isinstance(problem.data, JointSpectrumData):
        if not isinstance(problem.forward_model, JointForwardModel):
            return {}
        models = {
            model.name: model.forward_model
            for model in problem.forward_model.models
        }
        items = [
            (spectrum.name, spectrum.data, models[spectrum.name])
            for spectrum in problem.data.spectra
        ]
    else:
        items = [("spectrum", problem.data, problem.forward_model)]

    reconstructions = {}
    for name, data, forward_model in items:
        if not _supports_physical_reconstruction(
            data,
            problem.configuration,
            forward_model,
        ):
            continue
        evaluator = ReconstructionEvaluator(
            data,
            problem.configuration,
            forward_model=forward_model,
            batch_size=batch_size,
        )
        reconstructions[name] = evaluator.evaluate(free_values)
    return reconstructions


def _normalized_dynesty_weights(sampler):
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    return weights / np.sum(weights)


def run_inference(
    problem,
    backend,
    *,
    random_seed=0,
    progress=True,
    reconstruct=True,
    reconstruction_batch_size=None,
    **backend_options,
):
    """Run one problem with NUTS or dynesty and standardize its products."""
    if not isinstance(problem, InferenceProblem):
        raise TypeError("problem must be an InferenceProblem")
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if reconstruction_batch_size is not None and (
        not isinstance(reconstruction_batch_size, int)
        or reconstruction_batch_size < 1
    ):
        raise ValueError(
            "reconstruction_batch_size must be a positive integer or None"
        )
    backend = str(backend).lower()
    parameter_names = tuple(
        parameter.name
        for parameter in problem.configuration.free_parameters
    )

    if backend == "nuts":
        from lyapy.inference.numpyro_backend import run_nuts

        if "progress_bar" in backend_options:
            raise ValueError("pass progress instead of progress_bar")
        sampler = run_nuts(
            problem.data,
            problem.configuration,
            jax.random.key(random_seed),
            progress_bar=progress,
            forward_model=problem.forward_model,
            likelihood_function=problem.likelihood_function,
            **backend_options,
        )
        raw_samples = sampler.get_samples()
        samples = np.column_stack(
            [np.asarray(raw_samples[name]) for name in parameter_names]
        )
        weights = None
        diagnostics = summarize_nuts(sampler, parameter_names)
        log_evidence = None
        log_evidence_error = None
    elif backend == "dynesty":
        from lyapy.inference.dynesty_backend import run_dynesty

        if "print_progress" in backend_options:
            raise ValueError("pass progress instead of print_progress")
        sampler = run_dynesty(
            problem.data,
            problem.configuration,
            random_state=np.random.default_rng(random_seed),
            print_progress=progress,
            forward_model=problem.forward_model,
            likelihood_function=problem.likelihood_function,
            **backend_options,
        )
        samples = np.asarray(sampler.results.samples, dtype=np.float64)
        weights = _normalized_dynesty_weights(sampler)
        diagnostics = summarize_dynesty(sampler, parameter_names)
        log_evidence = float(sampler.results.logz[-1])
        log_evidence_error = float(sampler.results.logzerr[-1])
    else:
        raise ValueError("backend must be 'nuts' or 'dynesty'")

    reconstructions = (
        _evaluate_reconstructions(
            problem,
            samples,
            reconstruction_batch_size,
        )
        if reconstruct
        else {}
    )
    observations = _observation_mapping(problem)
    predictions = {
        name: reconstruction.convolved_observed
        for name, reconstruction in reconstructions.items()
    }
    if set(predictions) != set(observations):
        predictions = _evaluate_predictions(problem, samples)

    return InferenceResult(
        name=problem.name,
        backend=backend,
        parameter_names=parameter_names,
        samples=samples,
        weights=weights,
        diagnostics=diagnostics,
        log_evidence=log_evidence,
        log_evidence_error=log_evidence_error,
        problem_specification=problem.as_dict(),
        observations=observations,
        predictions=predictions,
        reconstructions=reconstructions,
    )


def _same_observations(first, second):
    first_observations = _observation_mapping(first)
    second_observations = _observation_mapping(second)
    if tuple(first_observations) != tuple(second_observations):
        return False
    for name, first_data in first_observations.items():
        second_data = second_observations[name]
        if not all(
            np.array_equal(
                getattr(first_data, array_name),
                getattr(second_data, array_name),
            )
            for array_name in (
                "wavelength",
                "flux",
                "error",
                "convolution_kernel",
            )
        ):
            return False
    return True


def compare_inference_problems(
    problems,
    *,
    random_seed=0,
    progress=True,
    reconstruct=True,
    reconstruction_batch_size=None,
    **dynesty_options,
):
    """Run dynesty for two or more matched problems and rank their evidence."""
    problems = tuple(problems)
    if len(problems) < 2:
        raise ValueError("at least two inference problems are required")
    if not all(isinstance(problem, InferenceProblem) for problem in problems):
        raise TypeError("problems must contain InferenceProblem values")
    names = [problem.name for problem in problems]
    if len(names) != len(set(names)):
        raise ValueError("inference problem names must be unique")
    reference = problems[0]
    if not all(
        _same_observations(reference, problem)
        for problem in problems[1:]
    ):
        raise ValueError("all inference problems must use identical observations")

    runs = []
    for index, problem in enumerate(problems):
        start = perf_counter()
        result = run_inference(
            problem,
            "dynesty",
            random_seed=random_seed + index,
            progress=progress,
            reconstruct=reconstruct,
            reconstruction_batch_size=reconstruction_batch_size,
            **dynesty_options,
        )
        runs.append(
            InferenceComparisonRun(
                result=result,
                runtime_seconds=perf_counter() - start,
            )
        )
    return InferenceComparisonResult(tuple(runs))


@dataclass(frozen=True)
class InferenceResult:
    """Portable posterior samples, predictions, diagnostics, and metadata."""

    name: str
    backend: str
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    weights: np.ndarray | None
    diagnostics: dict
    problem_specification: dict
    observations: dict[str, SpectrumData]
    predictions: dict[str, np.ndarray]
    reconstructions: dict[str, ReconstructionSamples] = field(default_factory=dict)
    log_evidence: float | None = None
    log_evidence_error: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("result name must not be empty")
        if self.backend not in {"nuts", "dynesty"}:
            raise ValueError("backend must be 'nuts' or 'dynesty'")
        samples = np.asarray(self.samples, dtype=np.float64)
        parameter_names = tuple(self.parameter_names)
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError("samples must have shape (draws, parameters)")
        if samples.shape[1] != len(parameter_names):
            raise ValueError("samples must match parameter_names")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must contain only finite values")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("parameter_names must be unique")
        weights = self.weights
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (samples.shape[0],):
                raise ValueError("weights must contain one value per draw")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
            if np.sum(weights) <= 0.0:
                raise ValueError("weights must have a positive sum")
            weights = weights / np.sum(weights)
        observations = dict(self.observations)
        if not all(
            isinstance(observation, SpectrumData)
            for observation in observations.values()
        ):
            raise TypeError("observations must contain SpectrumData values")
        predictions = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in self.predictions.items()
        }
        if not observations or set(predictions) != set(observations):
            raise ValueError("predictions must match all named observations")
        for name, prediction in predictions.items():
            expected = (samples.shape[0], observations[name].wavelength.size)
            if prediction.shape != expected:
                raise ValueError(
                    f"prediction {name!r} must have shape {expected}"
                )
        reconstructions = dict(self.reconstructions)
        if not set(reconstructions).issubset(observations):
            raise ValueError("reconstructions must match named observations")
        for name, reconstruction in reconstructions.items():
            if not isinstance(reconstruction, ReconstructionSamples):
                raise TypeError(
                    "reconstructions must contain ReconstructionSamples values"
                )
            model_wavelength = np.asarray(reconstruction.model_wavelength)
            intrinsic = np.asarray(reconstruction.intrinsic)
            convolved = np.asarray(reconstruction.convolved_observed)
            integrated_flux = np.asarray(
                reconstruction.integrated_intrinsic_flux
            )
            if model_wavelength.ndim != 1 or model_wavelength.size < 2:
                raise ValueError("model_wavelength must be one-dimensional")
            if intrinsic.shape != (samples.shape[0], model_wavelength.size):
                raise ValueError(
                    f"intrinsic reconstruction {name!r} has invalid shape"
                )
            if convolved.shape != predictions[name].shape:
                raise ValueError(
                    f"convolved reconstruction {name!r} has invalid shape"
                )
            if integrated_flux.shape != (samples.shape[0],):
                raise ValueError(
                    f"integrated-flux reconstruction {name!r} has invalid shape"
                )
            if not all(
                np.all(np.isfinite(values))
                for values in (
                    model_wavelength,
                    intrinsic,
                    convolved,
                    integrated_flux,
                )
            ):
                raise ValueError("reconstructions must contain finite values")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")
        if not isinstance(self.problem_specification, dict):
            raise TypeError("problem_specification must be a dictionary")
        evidence_values = (self.log_evidence, self.log_evidence_error)
        if (evidence_values[0] is None) != (evidence_values[1] is None):
            raise ValueError("evidence and its error must both be present")
        if evidence_values[0] is not None:
            if not all(math.isfinite(value) for value in evidence_values):
                raise ValueError("evidence values must be finite")
            if evidence_values[1] < 0.0:
                raise ValueError("log_evidence_error must be non-negative")
        if self.backend == "dynesty" and self.log_evidence is None:
            raise ValueError("dynesty results must contain evidence")
        if self.backend == "nuts" and self.log_evidence is not None:
            raise ValueError("NUTS results do not contain evidence")
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "reconstructions", reconstructions)

    @property
    def draws(self):
        return self.samples.shape[0]

    def parameter_samples(self):
        """Return posterior values keyed by free-parameter name."""
        return {
            name: self.samples[:, index]
            for index, name in enumerate(self.parameter_names)
        }

    def parameter_summary(self, interval_probability=0.9):
        """Return backend-neutral posterior summaries by parameter name."""
        return summarize_parameters(
            self.parameter_samples(),
            weights=self.weights,
            interval_probability=interval_probability,
        )

    def prediction_bands(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        """Summarize detector-space posterior predictions."""
        try:
            prediction = self.predictions[spectrum]
        except KeyError as error:
            raise KeyError(f"unknown spectrum {spectrum!r}") from error
        return posterior_percentile_bands(
            prediction,
            probabilities,
            weights=self.weights,
        )

    def posterior_predictive_bands(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
        random_seed=0,
    ):
        """Summarize replicated spectra including measurement uncertainty."""
        from lyapy.inference.result_diagnostics import (
            posterior_predictive_bands,
        )

        return posterior_predictive_bands(
            self,
            spectrum=spectrum,
            probabilities=probabilities,
            random_seed=random_seed,
        )

    def fit_diagnostics(
        self,
        spectrum="spectrum",
        *,
        random_seed=0,
        large_residual_threshold=3.0,
    ):
        """Return noise-aware fit diagnostics for one named spectrum."""
        from lyapy.inference.result_diagnostics import evaluate_fit_diagnostics

        return evaluate_fit_diagnostics(
            self,
            spectrum=spectrum,
            random_seed=random_seed,
            large_residual_threshold=large_residual_threshold,
        )

    def prior_boundary_diagnostics(
        self,
        *,
        boundary_fraction=0.05,
        warning_probability=0.2,
        boundary_distance_in_posterior_widths=0.5,
    ):
        """Report posterior mass near each bounded prior edge."""
        from lyapy.inference.prior import analyze_prior_boundaries

        return analyze_prior_boundaries(
            self,
            boundary_fraction=boundary_fraction,
            warning_probability=warning_probability,
            boundary_distance_in_posterior_widths=(
                boundary_distance_in_posterior_widths
            ),
        )

    def plot(
        self,
        spectrum="spectrum",
        *,
        random_seed=0,
        show_posterior_predictive=True,
    ):
        """Create the standard fit, residual, and reconstruction figure."""
        from lyapy.inference.result_diagnostics import plot_inference_result

        return plot_inference_result(
            self,
            spectrum=spectrum,
            random_seed=random_seed,
            show_posterior_predictive=show_posterior_predictive,
        )

    def integrated_flux_summary(
        self,
        spectrum="spectrum",
        probabilities=DEFAULT_PERCENTILE_PROBABILITIES,
    ):
        """Summarize intrinsic flux when the forward model exposes it."""
        try:
            reconstruction = self.reconstructions[spectrum]
        except KeyError as error:
            raise ValueError(
                f"spectrum {spectrum!r} has no intrinsic reconstruction"
            ) from error
        return summarize_integrated_flux(
            reconstruction.integrated_intrinsic_flux,
            probabilities,
            weights=self.weights,
        )

    def as_dict(self):
        """Return metadata and summaries while excluding numerical arrays."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "name": self.name,
            "backend": self.backend,
            "parameter_names": list(self.parameter_names),
            "draws": self.draws,
            "weighted": self.weights is not None,
            "log_evidence": self.log_evidence,
            "log_evidence_error": self.log_evidence_error,
            "diagnostics": self.diagnostics,
            "problem": self.problem_specification,
            "spectra": {
                name: {
                    **_spectrum_specification(observation),
                    "has_intrinsic_reconstruction": name in self.reconstructions,
                }
                for name, observation in self.observations.items()
            },
        }

    def save(self, path):
        """Write JSON metadata plus compressed arrays without pickle."""
        json_path = Path(path)
        if json_path.suffix != ".json":
            json_path = json_path.with_suffix(".json")
        arrays_path = json_path.with_suffix(".npz")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"posterior_samples": self.samples}
        if self.weights is not None:
            arrays["posterior_weights"] = self.weights
        spectra = []
        for index, (name, observation) in enumerate(self.observations.items()):
            prefix = f"spectrum_{index}"
            arrays[f"{prefix}_wavelength"] = observation.wavelength
            arrays[f"{prefix}_flux"] = observation.flux
            arrays[f"{prefix}_error"] = observation.error
            arrays[f"{prefix}_convolution_kernel"] = (
                observation.convolution_kernel
            )
            arrays[f"{prefix}_predictions"] = self.predictions[name]
            reconstruction = self.reconstructions.get(name)
            if reconstruction is not None:
                arrays[f"{prefix}_model_wavelength"] = (
                    reconstruction.model_wavelength
                )
                arrays[f"{prefix}_intrinsic"] = reconstruction.intrinsic
                arrays[f"{prefix}_integrated_intrinsic_flux"] = (
                    reconstruction.integrated_intrinsic_flux
                )
            spectra.append(
                {
                    "name": name,
                    "array_prefix": prefix,
                    "has_intrinsic_reconstruction": reconstruction is not None,
                }
            )
        np.savez_compressed(arrays_path, **arrays)
        metadata = self.as_dict()
        metadata.update(arrays_file=arrays_path.name, array_spectra=spectra)
        json_path.write_text(
            json.dumps(_json_compatible(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        return json_path, arrays_path


@dataclass(frozen=True)
class InferenceComparisonRun:
    """One standardized dynesty result and its elapsed runtime."""

    result: InferenceResult
    runtime_seconds: float

    def __post_init__(self):
        if not isinstance(self.result, InferenceResult):
            raise TypeError("result must be an InferenceResult")
        if self.result.backend != "dynesty":
            raise ValueError("model comparison requires dynesty results")
        if (
            not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0.0
        ):
            raise ValueError("runtime_seconds must be finite and non-negative")

    @property
    def evidence(self):
        return EvidenceEstimate(
            self.result.name,
            self.result.log_evidence,
            self.result.log_evidence_error,
        )


@dataclass(frozen=True)
class InferenceComparisonResult:
    """Portable posterior products and evidence ranking for several models."""

    runs: tuple[InferenceComparisonRun, ...]

    def __post_init__(self):
        runs = tuple(self.runs)
        if len(runs) < 2:
            raise ValueError("at least two comparison runs are required")
        if not all(isinstance(run, InferenceComparisonRun) for run in runs):
            raise TypeError("runs must contain InferenceComparisonRun values")
        names = [run.result.name for run in runs]
        if len(names) != len(set(names)):
            raise ValueError("comparison result names must be unique")
        object.__setattr__(self, "runs", runs)

    @property
    def ranking(self):
        """Return models ordered from highest to lowest evidence."""
        return rank_evidence(run.evidence for run in self.runs)

    @property
    def best_model(self):
        return self.ranking[0].model

    def get_result(self, model):
        """Return the standardized inference result for one named model."""
        for run in self.runs:
            if run.result.name == model:
                return run.result
        raise KeyError(model)

    def compare(self, numerator, denominator):
        """Return the evidence difference for two named models."""
        estimates = {run.result.name: run.evidence for run in self.runs}
        try:
            return compare_evidence(
                estimates[numerator],
                estimates[denominator],
            )
        except KeyError as error:
            raise KeyError(f"unknown model {error.args[0]!r}") from error

    def model_average(self, model_prior_probabilities=None):
        """Combine posterior products using evidence-based model weights."""
        from lyapy.inference.averaging import model_average

        return model_average(
            self,
            model_prior_probabilities=model_prior_probabilities,
        )

    def plot(
        self,
        spectrum="spectrum",
        *,
        random_seed=0,
        show_68_percent_bands=True,
    ):
        """Create a detector-space comparison and residual figure."""
        from lyapy.inference.result_diagnostics import (
            plot_inference_comparison,
        )

        return plot_inference_comparison(
            self,
            spectrum=spectrum,
            random_seed=random_seed,
            show_68_percent_bands=show_68_percent_bands,
        )

    def as_dict(self):
        """Return JSON-compatible rankings and per-model result metadata."""
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "best_model": self.best_model,
            "ranking": [asdict(item) for item in self.ranking],
            "models": {
                run.result.name: {
                    "runtime_seconds": run.runtime_seconds,
                    "result": run.result.as_dict(),
                }
                for run in self.runs
            },
        }

    def save(self, directory):
        """Save a comparison manifest plus every model's portable result."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = self.as_dict()
        result_paths = {}
        for index, run in enumerate(self.runs):
            json_path, arrays_path = run.result.save(
                directory / f"model_{index}"
            )
            result_paths[run.result.name] = (json_path, arrays_path)
            metadata["models"][run.result.name]["result_file"] = (
                json_path.name
            )
        manifest_path = directory / "comparison.json"
        manifest_path.write_text(
            json.dumps(_json_compatible(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path, result_paths


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_inference_result(path):
    """Load a result written by :meth:`InferenceResult.save`."""
    json_path = Path(path)
    if json_path.suffix != ".json":
        json_path = json_path.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported inference-result schema version")
    arrays_path = json_path.parent / metadata["arrays_file"]
    observations = {}
    predictions = {}
    reconstructions = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        samples = np.asarray(arrays["posterior_samples"])
        weights = (
            np.asarray(arrays["posterior_weights"])
            if "posterior_weights" in arrays
            else None
        )
        for spectrum in metadata["array_spectra"]:
            name = spectrum["name"]
            prefix = spectrum["array_prefix"]
            observation = SpectrumData(
                arrays[f"{prefix}_wavelength"],
                arrays[f"{prefix}_flux"],
                arrays[f"{prefix}_error"],
                arrays[f"{prefix}_convolution_kernel"],
            )
            observations[name] = observation
            predictions[name] = np.asarray(arrays[f"{prefix}_predictions"])
            if spectrum["has_intrinsic_reconstruction"]:
                reconstructions[name] = ReconstructionSamples(
                    model_wavelength=np.asarray(
                        arrays[f"{prefix}_model_wavelength"]
                    ),
                    intrinsic=np.asarray(arrays[f"{prefix}_intrinsic"]),
                    convolved_observed=predictions[name],
                    integrated_intrinsic_flux=np.asarray(
                        arrays[f"{prefix}_integrated_intrinsic_flux"]
                    ),
                )
    return InferenceResult(
        name=metadata["name"],
        backend=metadata["backend"],
        parameter_names=tuple(metadata["parameter_names"]),
        samples=samples,
        weights=weights,
        diagnostics=metadata["diagnostics"],
        log_evidence=metadata["log_evidence"],
        log_evidence_error=metadata["log_evidence_error"],
        problem_specification=metadata["problem"],
        observations=observations,
        predictions=predictions,
        reconstructions=reconstructions,
    )


def load_inference_comparison(path):
    """Load a comparison written by :meth:`InferenceComparisonResult.save`."""
    manifest_path = Path(path)
    if manifest_path.suffix != ".json":
        manifest_path = manifest_path / "comparison.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        raise ValueError("unsupported inference-comparison schema version")
    runs = []
    for name, model in metadata["models"].items():
        result = load_inference_result(
            manifest_path.parent / model["result_file"]
        )
        if result.name != name:
            raise ValueError("comparison manifest and result names must match")
        runs.append(
            InferenceComparisonRun(
                result=result,
                runtime_seconds=model["runtime_seconds"],
            )
        )
    comparison = InferenceComparisonResult(tuple(runs))
    if comparison.best_model != metadata["best_model"]:
        raise ValueError("comparison manifest contains an inconsistent ranking")
    return comparison
