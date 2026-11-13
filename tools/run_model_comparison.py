"""Compare self-reversed and non-reversed models on synthetic GJ 411 data."""

import argparse
import json
from pathlib import Path
import sys

try:
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing optional inference dependency {error.name!r}.\n\n"
        "From the repository root, install the validated inference stack with:\n\n"
        "    python -m pip install -e '.[inference]'\n\n"
        "Then rerun this command."
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import make_reversal_comparison_configurations
from lyapy.inference.comparison import (
    ModelHypothesis,
    compare_evidence,
    run_evidence_comparison,
)
from lyapy.inference.diagnostics import summarize_dynesty
from run_inference_validation import make_synthetic_problem, predict


def posterior_mean(sampler):
    """Return the dynesty-weighted mean parameter vector."""
    weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
    weights = weights / np.sum(weights)
    return np.average(sampler.results.samples, axis=0, weights=weights)


def make_plot(
    data,
    reversal_configuration,
    no_reversal_configuration,
    reversal_sampler,
    no_reversal_sampler,
    comparison,
    data_label="synthetic data",
    title_prefix="Self-reversal comparison",
):
    """Plot the two posterior-mean curves and their normalized residuals."""
    reversal_model = predict(
        data,
        reversal_configuration,
        posterior_mean(reversal_sampler),
    )
    no_reversal_model = predict(
        data,
        no_reversal_configuration,
        posterior_mean(no_reversal_sampler),
    )
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    profile_axis.errorbar(
        data.wavelength,
        data.flux * 1e12,
        yerr=data.error * 1e12,
        color="0.65",
        linewidth=0.8,
        label=data_label,
    )
    profile_axis.plot(
        data.wavelength,
        reversal_model * 1e12,
        color="#d6278b",
        linewidth=2.0,
        label="self reversal",
    )
    profile_axis.plot(
        data.wavelength,
        no_reversal_model * 1e12,
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label="no reversal",
    )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False)
    profile_axis.grid(alpha=0.2)

    residual_axis.axhline(0.0, color="0.4", linewidth=1.0)
    residual_axis.plot(
        data.wavelength,
        (data.flux - reversal_model) / data.error,
        color="#d6278b",
        label="self reversal",
    )
    residual_axis.plot(
        data.wavelength,
        (data.flux - no_reversal_model) / data.error,
        color="#1f77b4",
        linestyle="--",
        label="no reversal",
    )
    residual_axis.set_xlabel(r"wavelength ($\AA$)")
    residual_axis.set_ylabel("residual / error")
    residual_axis.grid(alpha=0.2)
    figure.suptitle(
        f"{title_prefix}: Δlog Z = "
        f"{comparison.delta_log_evidence:.1f} ± "
        f"{comparison.delta_log_evidence_error:.1f}"
    )
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlive", type=int, default=300)
    parser.add_argument("--dlogz", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_comparison.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON destination (defaults to the plot name with a .json suffix)",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    report_path = arguments.report or arguments.output.with_suffix(".json")
    data, original_configuration, truth = make_synthetic_problem(arguments.seed)
    reversal_configuration, no_reversal_configuration = (
        make_reversal_comparison_configurations(
            original_configuration,
            reversal_lower=1.0,
            reversal_upper=1.9,
        )
    )

    selection = run_evidence_comparison(
        data,
        (
            ModelHypothesis("self reversal", reversal_configuration),
            ModelHypothesis("no reversal", no_reversal_configuration),
        ),
        random_seed=arguments.seed,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        print_progress=not arguments.no_progress,
    )
    reversal_run = selection.get_run("self reversal")
    no_reversal_run = selection.get_run("no reversal")
    reversal_sampler = reversal_run.sampler
    no_reversal_sampler = no_reversal_run.sampler
    comparison = compare_evidence(
        reversal_run.evidence,
        no_reversal_run.evidence,
    )
    reversal_summary = summarize_dynesty(
        reversal_sampler,
        ("log_amplitude", "reversal_depth"),
        truth=truth,
    )
    no_reversal_summary = summarize_dynesty(
        no_reversal_sampler,
        ("log_amplitude",),
        truth={"log_amplitude": truth["log_amplitude"]},
    )
    report = {
        "problem": "synthetic GJ 411 self-reversal model comparison",
        "seed": arguments.seed,
        "generating_model": "self reversal",
        "settings": {
            "live_points": arguments.nlive,
            "stopping_log_evidence": arguments.dlogz,
        },
        "runtime_seconds": {
            "self_reversal": reversal_run.runtime_seconds,
            "no_reversal": no_reversal_run.runtime_seconds,
        },
        "model_selection": selection.as_dict(),
        "comparison": comparison.as_dict(),
        "self_reversal": reversal_summary,
        "no_reversal": no_reversal_summary,
    }

    figure = make_plot(
        data,
        reversal_configuration,
        no_reversal_configuration,
        reversal_sampler,
        no_reversal_sampler,
        comparison,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["comparison"], indent=2))
    print(f"Self-reversal runtime: {reversal_run.runtime_seconds:.2f} s")
    print(f"No-reversal runtime:   {no_reversal_run.runtime_seconds:.2f} s")
    print(f"Wrote {arguments.output.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
