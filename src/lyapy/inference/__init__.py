"""Optional inference adapters for the differentiable lyapy model."""

from .common import (
    GJ411_CONFIGURATION,
    GJ411_STELLAR_CONFIGURATION,
    InferenceConfiguration,
    SpectrumData,
    TruncatedNormalParameter,
    UniformParameter,
    make_reversal_comparison_configurations,
)
from .diagnostics import (
    summarize_dynesty,
    summarize_nuts,
    summarize_parameters,
    weighted_quantile,
)
from .comparison import (
    EvidenceEstimate,
    EvidenceRanking,
    ModelComparison,
    ModelEvidenceRun,
    ModelHypothesis,
    ModelSelectionResult,
    compare_evidence,
    rank_evidence,
    run_evidence_comparison,
)
from .composable import (
    COMPOSABLE_GJ411_BASE_PARAMETERS,
    COMPOSABLE_GJ411_PARAMETER_NAMES,
    GJ411_COMPOSABLE_CONFIGURATION,
    ComposableInferenceProblem,
    build_gj411_composable_forward_model,
    build_gj411_observation_operator,
    load_gj411_composable_problem,
)
from .identifiability import (
    IdentifiabilityResult,
    analyze_local_identifiability,
    prior_standard_deviation,
)
from .model_catalog import (
    ModelCatalogEntry,
    build_gj411_model_catalog,
    make_model_catalog_entry,
)
from .reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    ReconstructionSamples,
    evaluate_reconstruction_samples,
    posterior_percentile_bands,
    summarize_integrated_flux,
)

__all__ = [
    "GJ411_CONFIGURATION",
    "GJ411_COMPOSABLE_CONFIGURATION",
    "GJ411_STELLAR_CONFIGURATION",
    "COMPOSABLE_GJ411_BASE_PARAMETERS",
    "COMPOSABLE_GJ411_PARAMETER_NAMES",
    "DEFAULT_PERCENTILE_PROBABILITIES",
    "EvidenceEstimate",
    "EvidenceRanking",
    "InferenceConfiguration",
    "ComposableInferenceProblem",
    "IdentifiabilityResult",
    "ModelComparison",
    "ModelCatalogEntry",
    "ModelEvidenceRun",
    "ModelHypothesis",
    "ModelSelectionResult",
    "ReconstructionSamples",
    "SpectrumData",
    "TruncatedNormalParameter",
    "UniformParameter",
    "summarize_dynesty",
    "summarize_nuts",
    "summarize_parameters",
    "weighted_quantile",
    "make_reversal_comparison_configurations",
    "evaluate_reconstruction_samples",
    "posterior_percentile_bands",
    "summarize_integrated_flux",
    "compare_evidence",
    "rank_evidence",
    "run_evidence_comparison",
    "analyze_local_identifiability",
    "prior_standard_deviation",
    "build_gj411_composable_forward_model",
    "build_gj411_observation_operator",
    "build_gj411_model_catalog",
    "load_gj411_composable_problem",
    "make_model_catalog_entry",
]
