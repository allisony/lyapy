"""Evidence summaries and pairwise Bayesian model comparison."""

from dataclasses import asdict, dataclass
import math


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
