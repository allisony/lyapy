"""Evidence summaries and reusable Bayesian model comparison."""

from dataclasses import asdict, dataclass
import math
from time import perf_counter

import numpy as np

from lyapy.jax_models import evaluate_voigt_lya_model_jax
from lyapy.likelihoods import gaussian_log_likelihood


@dataclass(frozen=True)
class EvidenceEstimate:
    """One model's nested-sampling evidence estimate."""

    model: str
    log_evidence: float
    log_evidence_error: float

    def __post_init__(self):
        if not self.model:
            raise ValueError("model name must not be empty")
        if not math.isfinite(self.log_evidence):
            raise ValueError("log evidence must be finite")
        if (
            not math.isfinite(self.log_evidence_error)
            or self.log_evidence_error < 0.0
        ):
            raise ValueError("log-evidence error must be finite and non-negative")

    @classmethod
    def from_dynesty(cls, model, sampler):
        """Build an estimate from a completed dynesty sampler."""
        return cls(
            model=model,
            log_evidence=float(sampler.results.logz[-1]),
            log_evidence_error=float(sampler.results.logzerr[-1]),
        )


@dataclass(frozen=True)
class ModelComparison:
    """Evidence difference for a numerator model versus a denominator model."""

    numerator: EvidenceEstimate
    denominator: EvidenceEstimate
    delta_log_evidence: float
    delta_log_evidence_error: float
    log10_bayes_factor: float
    favored_model: str
    strength: str

    def as_dict(self):
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class ModelHypothesis:
    """Named parameter configuration representing one scientific model."""

    name: str
    configuration: object
    forward_model: object = evaluate_voigt_lya_model_jax
    likelihood_function: object = gaussian_log_likelihood

    def __post_init__(self):
        if not self.name:
            raise ValueError("model name must not be empty")
        if not hasattr(self.configuration, "free_parameters"):
            raise TypeError("configuration must define free_parameters")
        if not callable(self.forward_model):
            raise TypeError("forward_model must be callable")
        if not callable(self.likelihood_function):
            raise TypeError("likelihood_function must be callable")

    def as_dict(self):
        """Return a reproducible JSON-compatible model specification."""
        parameters = []
        for parameter in self.configuration.free_parameters:
            specification = asdict(parameter)
            specification["prior"] = type(parameter).__name__
            parameters.append(specification)
        result = {
            "name": self.name,
            "forward_model": _callable_name(self.forward_model),
            "likelihood_function": _callable_name(self.likelihood_function),
            "parameter_names": list(self.configuration.parameter_names),
            "base_parameters": list(self.configuration.base_parameters),
            "free_parameters": parameters,
        }
        if hasattr(self.likelihood_function, "as_dict"):
            result["likelihood_specification"] = (
                self.likelihood_function.as_dict()
            )
        if hasattr(self.forward_model, "as_dict"):
            result["forward_model_specification"] = self.forward_model.as_dict()
        return result


@dataclass(frozen=True)
class ModelEvidenceRun:
    """Completed evidence calculation for one hypothesis."""

    hypothesis: ModelHypothesis
    evidence: EvidenceEstimate
    runtime_seconds: float
    sampler: object

    def __post_init__(self):
        if self.evidence.model != self.hypothesis.name:
            raise ValueError("evidence and hypothesis model names must match")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be finite and non-negative")

    def as_dict(self):
        """Return report fields while excluding the live sampler object."""
        return {
            "hypothesis": self.hypothesis.as_dict(),
            "evidence": asdict(self.evidence),
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass(frozen=True)
class EvidenceRanking:
    """One model ranked relative to the highest-evidence model."""

    rank: int
    model: str
    log_evidence: float
    log_evidence_error: float
    delta_log_evidence_below_best: float
    delta_log_evidence_error: float
    log10_bayes_factor_for_best: float
    strength: str


@dataclass(frozen=True)
class ModelSelectionResult:
    """Evidence runs and ranking for two or more hypotheses."""

    runs: tuple[ModelEvidenceRun, ...]
    ranking: tuple[EvidenceRanking, ...]

    @property
    def best_model(self):
        return self.ranking[0].model

    def get_run(self, model):
        """Return one named model run."""
        for run in self.runs:
            if run.hypothesis.name == model:
                return run
        raise KeyError(model)

    def as_dict(self):
        """Return a JSON-compatible multi-model report."""
        return {
            "best_model": self.best_model,
            "ranking": [asdict(item) for item in self.ranking],
            "models": {
                run.hypothesis.name: run.as_dict()
                for run in self.runs
            },
        }


def evidence_strength(delta_log_evidence):
    """Return a conservative descriptive label for an absolute log Bayes factor."""
    magnitude = abs(delta_log_evidence)
    if magnitude < 1.0:
        return "inconclusive"
    if magnitude < 2.5:
        return "weak"
    if magnitude < 5.0:
        return "moderate"
    return "strong"


def _callable_name(function):
    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


def compare_evidence(numerator, denominator):
    """Compare two independent log-evidence estimates."""
    delta = numerator.log_evidence - denominator.log_evidence
    error = math.hypot(
        numerator.log_evidence_error,
        denominator.log_evidence_error,
    )
    favored_model = numerator.model if delta >= 0.0 else denominator.model
    return ModelComparison(
        numerator=numerator,
        denominator=denominator,
        delta_log_evidence=delta,
        delta_log_evidence_error=error,
        log10_bayes_factor=delta / math.log(10.0),
        favored_model=favored_model,
        strength=evidence_strength(delta),
    )


def rank_evidence(estimates):
    """Rank two or more evidence estimates relative to the best model."""
    estimates = tuple(estimates)
    if len(estimates) < 2:
        raise ValueError("at least two evidence estimates are required")
    names = [estimate.model for estimate in estimates]
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique")
    ordered = sorted(estimates, key=lambda estimate: estimate.log_evidence, reverse=True)
    best = ordered[0]
    ranking = []
    for rank, estimate in enumerate(ordered, start=1):
        delta = best.log_evidence - estimate.log_evidence
        error = (
            0.0
            if estimate is best
            else math.hypot(
                best.log_evidence_error,
                estimate.log_evidence_error,
            )
        )
        ranking.append(
            EvidenceRanking(
                rank=rank,
                model=estimate.model,
                log_evidence=estimate.log_evidence,
                log_evidence_error=estimate.log_evidence_error,
                delta_log_evidence_below_best=delta,
                delta_log_evidence_error=error,
                log10_bayes_factor_for_best=delta / math.log(10.0),
                strength=(
                    "reference" if estimate is best else evidence_strength(delta)
                ),
            )
        )
    return tuple(ranking)


def run_evidence_comparison(
    data,
    hypotheses,
    *,
    random_seed=0,
    sampler_runner=None,
    **sampler_options,
):
    """Run and rank any set of model configurations with dynesty.

    ``sampler_runner`` is injectable for testing or alternative nested-sampling
    adapters. It must accept ``data``, ``configuration``, and ``random_state``.
    """
    hypotheses = tuple(hypotheses)
    if len(hypotheses) < 2:
        raise ValueError("at least two model hypotheses are required")
    names = [hypothesis.name for hypothesis in hypotheses]
    if len(names) != len(set(names)):
        raise ValueError("model hypothesis names must be unique")
    if sampler_runner is None:
        from lyapy.inference.dynesty_backend import run_dynesty

        sampler_runner = run_dynesty

    runs = []
    for index, hypothesis in enumerate(hypotheses):
        start = perf_counter()
        sampler = sampler_runner(
            data,
            hypothesis.configuration,
            random_state=np.random.default_rng(random_seed + index),
            forward_model=hypothesis.forward_model,
            likelihood_function=hypothesis.likelihood_function,
            **sampler_options,
        )
        runtime = perf_counter() - start
        evidence = EvidenceEstimate.from_dynesty(hypothesis.name, sampler)
        runs.append(
            ModelEvidenceRun(
                hypothesis=hypothesis,
                evidence=evidence,
                runtime_seconds=runtime,
                sampler=sampler,
            )
        )
    runs = tuple(runs)
    return ModelSelectionResult(
        runs=runs,
        ranking=rank_evidence(run.evidence for run in runs),
    )
