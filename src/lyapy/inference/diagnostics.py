"""Backend-independent posterior summaries for lyapy inference runs."""

import numpy as np


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def weighted_quantile(values, quantiles, weights=None):
    """Return quantiles for one-dimensional samples with optional weights."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie between zero and one")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite entries")
    if weights is None:
        return np.quantile(values, quantiles)

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError("weights must match the values shape")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total_weight = np.sum(weights)
    if total_weight <= 0.0:
        raise ValueError("weights must have a positive sum")

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order] / total_weight
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return np.interp(
        quantiles,
        cumulative,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def summarize_parameters(
    samples,
    truth=None,
    weights=None,
    interval_probability=0.9,
):
    """Summarize named posterior samples using a central credible interval."""
    if not 0.0 < interval_probability < 1.0:
        raise ValueError("interval_probability must lie strictly between zero and one")
    truth = {} if truth is None else truth
    tail = 0.5 * (1.0 - interval_probability)
    quantiles = (tail, 0.5, 1.0 - tail)
    summaries = {}
    for name, sample in samples.items():
        values = np.asarray(sample, dtype=np.float64).reshape(-1)
        lower, median, upper = weighted_quantile(values, quantiles, weights)
        if weights is None:
            mean = np.mean(values)
            standard_deviation = np.std(values)
        else:
            normalized_weights = np.asarray(weights, dtype=np.float64)
            normalized_weights = normalized_weights / np.sum(normalized_weights)
            mean = np.sum(normalized_weights * values)
            standard_deviation = np.sqrt(
                np.sum(normalized_weights * (values - mean) ** 2)
            )
        parameter_truth = truth.get(name)
        summaries[name] = {
            "mean": float(mean),
            "standard_deviation": float(standard_deviation),
            "median": float(median),
            "credible_interval": [float(lower), float(upper)],
            "truth": None if parameter_truth is None else float(parameter_truth),
            "truth_in_interval": (
                None
                if parameter_truth is None
                else bool(lower <= parameter_truth <= upper)
            ),
        }
    return summaries


def summarize_nuts(
    sampler,
    parameter_names,
    truth=None,
    interval_probability=0.9,
):
    """Summarize a NumPyro MCMC run, including convergence diagnostics."""
    from numpyro.diagnostics import summary

    chain_samples = sampler.get_samples(group_by_chain=True)
    parameter_samples = {name: chain_samples[name] for name in parameter_names}
    diagnostic_summary = summary(
        parameter_samples,
        prob=interval_probability,
        group_by_chain=True,
    )
    parameter_summary = summarize_parameters(
        parameter_samples,
        truth=truth,
        interval_probability=interval_probability,
    )
    for name, diagnostics in diagnostic_summary.items():
        parameter_summary[name]["effective_sample_size"] = _finite_float(
            diagnostics["n_eff"]
        )
        parameter_summary[name]["r_hat"] = _finite_float(diagnostics["r_hat"])

    extra_fields = sampler.get_extra_fields(group_by_chain=True)
    divergences = np.asarray(extra_fields["diverging"])
    acceptance_probability = np.asarray(extra_fields["accept_prob"])
    tree_steps = np.asarray(extra_fields["num_steps"])
    return {
        "parameters": parameter_summary,
        "chains": int(next(iter(parameter_samples.values())).shape[0]),
        "draws_per_chain": int(next(iter(parameter_samples.values())).shape[1]),
        "divergences": int(np.sum(divergences)),
        "mean_acceptance_probability": float(np.mean(acceptance_probability)),
        "mean_tree_steps": float(np.mean(tree_steps)),
    }


def summarize_dynesty(
    sampler,
    parameter_names,
    truth=None,
    interval_probability=0.9,
):
    """Summarize a dynesty run using its normalized importance weights."""
    results = sampler.results
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / np.sum(weights)
    samples = {
        name: results.samples[:, index]
        for index, name in enumerate(parameter_names)
    }
    return {
        "parameters": summarize_parameters(
            samples,
            truth=truth,
            weights=weights,
            interval_probability=interval_probability,
        ),
        "posterior_effective_sample_size": float(1.0 / np.sum(weights**2)),
        "likelihood_calls": int(np.sum(results.ncall)),
        "sampling_efficiency_percent": float(results.eff),
        "log_evidence": float(results.logz[-1]),
        "log_evidence_error": float(results.logzerr[-1]),
    }
