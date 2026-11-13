"""Compare self-reversed and non-reversed models for observed GJ 411."""

import argparse
import json
from pathlib import Path
import sys

try:
    import dynesty  # noqa: F401 -- fail early with an actionable message
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. From the repository root, run:\n\n"
        "    python -m pip install -e '.[inference]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import (
    GJ411_CONFIGURATION,
    SpectrumData,
    make_reversal_comparison_configurations,
)
from lyapy.inference.comparison import (
    ModelHypothesis,
    compare_evidence,
    run_evidence_comparison,
)
from lyapy.inference.diagnostics import summarize_dynesty
from lyapy.observations import load_gj411_observation
from run_model_comparison import make_plot, posterior_mean
from run_stellar_recovery import predict


def load_problem(reversal_upper=1.8):
    """Return the reviewed GJ 411 data and matched reversal hypotheses."""
    observation = load_gj411_observation()
    data = SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
    )
    self_reversed, no_reversal = make_reversal_comparison_configurations(
        GJ411_CONFIGURATION,
        reversal_lower=0.0,
        reversal_upper=reversal_upper,
    )
    return data, self_reversed, no_reversal


def fit_summary(data, configuration, sampler):
    """Return residual statistics for the posterior-mean parameters."""
    model = predict(data, configuration, posterior_mean(sampler))
    residuals = (data.flux - model) / data.error
    chi_squared = np.sum(residuals**2)
    degrees_of_freedom = data.wavelength.size - configuration.ndim
    return {
        "chi_squared": float(chi_squared),
        "degrees_of_freedom": int(degrees_of_freedom),
        "reduced_chi_squared": float(chi_squared / degrees_of_freedom),
        "residual_root_mean_square": float(np.sqrt(np.mean(residuals**2))),
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--dlogz", type=float, default=0.05)
    parser.add_argument(
        "--dynesty-sample",
        choices=("rwalk", "rslice"),
        default="rwalk",
    )
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--reversal-max", type=float, default=1.8)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("gj411_model_comparison"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    data, self_reversed, no_reversal = load_problem(arguments.reversal_max)
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")

    selection = run_evidence_comparison(
        data,
        (
            ModelHypothesis("self reversal", self_reversed),
            ModelHypothesis("no reversal", no_reversal),
        ),
        random_seed=arguments.seed,
        nlive=arguments.nlive,
        dlogz=arguments.dlogz,
        print_progress=not arguments.no_progress,
        sample=arguments.dynesty_sample,
        bootstrap=0,
    )
    reversal_run = selection.get_run("self reversal")
    no_reversal_run = selection.get_run("no reversal")
    reversal_sampler = reversal_run.sampler
    no_reversal_sampler = no_reversal_run.sampler
    comparison = compare_evidence(
        reversal_run.evidence,
        no_reversal_run.evidence,
    )
    reversal_names = tuple(
        parameter.name for parameter in self_reversed.free_parameters
    )
    no_reversal_names = tuple(
        parameter.name for parameter in no_reversal.free_parameters
    )
    report = {
        "problem": "observed GJ 411 self-reversal model comparison",
        "seed": arguments.seed,
        "hypotheses": {
            "self_reversal": {
                "reversal_depth_prior": [0.0, arguments.reversal_max],
                "free_parameters": list(reversal_names),
            },
            "no_reversal": {
                "reversal_depth": 0.0,
                "free_parameters": list(no_reversal_names),
            },
        },
        "settings": {
            "live_points": arguments.nlive,
            "stopping_log_evidence": arguments.dlogz,
            "dynesty_sample": arguments.dynesty_sample,
        },
        "runtime_seconds": {
            "self_reversal": reversal_run.runtime_seconds,
            "no_reversal": no_reversal_run.runtime_seconds,
        },
        "model_selection": selection.as_dict(),
        "comparison": comparison.as_dict(),
        "self_reversal": {
            "posterior": summarize_dynesty(
                reversal_sampler,
                reversal_names,
            ),
            "fit": fit_summary(data, self_reversed, reversal_sampler),
        },
        "no_reversal": {
            "posterior": summarize_dynesty(
                no_reversal_sampler,
                no_reversal_names,
            ),
            "fit": fit_summary(data, no_reversal, no_reversal_sampler),
        },
    }
    figure = make_plot(
        data,
        self_reversed,
        no_reversal,
        reversal_sampler,
        no_reversal_sampler,
        comparison,
        data_label="GJ 411 STIS data",
        title_prefix="Observed GJ 411 self-reversal comparison",
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["comparison"], indent=2))
    print(f"Self-reversal runtime: {reversal_run.runtime_seconds:.2f} s")
    print(f"No-reversal runtime:   {no_reversal_run.runtime_seconds:.2f} s")
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
