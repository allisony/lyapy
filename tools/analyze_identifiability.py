"""Analyze local identifiability of the eight free GJ 411 parameters."""

import argparse
import json
from pathlib import Path
import sys

try:
    import jax
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing optional JAX dependency. From the repository root, run:\n\n"
        "    python -m pip install -e '.[inference]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import GJ411_CONFIGURATION, SpectrumData
from lyapy.inference.identifiability import analyze_local_identifiability
from lyapy.observations import load_gj411_observation


PLOT_LABELS = {
    "stellar_velocity": "stellar velocity",
    "log_amplitude": "log amplitude",
    "lorentz_fwhm": "Lorentz FWHM",
    "gaussian_fwhm": "Gaussian FWHM",
    "h1_column": "H I column",
    "h1_doppler_b": "H I Doppler b",
    "h1_velocity": "H I velocity",
    "reversal_depth": "reversal depth",
}


def make_plot(result):
    """Visualize singular information, shrinkage, and local correlations."""
    labels = [PLOT_LABELS.get(name, name) for name in result.parameter_names]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].semilogy(
        np.arange(1, len(result.singular_values) + 1),
        result.singular_values,
        marker="o",
        color="#d6278b",
    )
    axes[0].set_xlabel("local parameter direction")
    axes[0].set_ylabel("error-weighted singular value")
    axes[0].set_title("Data information by direction")
    axes[0].grid(alpha=0.2)

    positions = np.arange(len(labels))
    axes[1].barh(
        positions,
        result.posterior_to_prior_scale,
        color="#1f77b4",
    )
    axes[1].set_yticks(positions, labels=labels)
    axes[1].invert_yaxis()
    axes[1].axvline(1.0, color="0.3", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(0.0, 1.05)
    axes[1].set_xlabel("local posterior σ / prior σ")
    axes[1].set_title("Approximate prior-to-posterior shrinkage")
    axes[1].grid(axis="x", alpha=0.2)

    image = axes[2].imshow(
        result.posterior_correlation,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    axes[2].set_xticks(positions, labels=labels, rotation=60, ha="right")
    axes[2].set_yticks(positions, labels=labels)
    axes[2].set_title("Local posterior correlation")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle("GJ 411 local parameter identifiability")
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("identifiability.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON destination (defaults to the plot name with a .json suffix)",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    report_path = arguments.report or arguments.output.with_suffix(".json")
    observation = load_gj411_observation()
    data = SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
    )
    result = analyze_local_identifiability(data, GJ411_CONFIGURATION)
    report = result.as_dict()
    report["target"] = observation.name
    report["wavelength_samples"] = int(observation.wavelength.size)

    figure = make_plot(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Numerical rank:   {result.numerical_rank}/{len(result.parameter_names)}")
    print(f"Condition number: {result.condition_number:.3g}")
    print("Weakest direction:")
    for item in result.weakest_direction()[:4]:
        print(f"  {item['parameter']:<20} {item['loading']:+.3f}")
    correlations = result.strongest_correlations()
    print("Strong correlations:")
    if correlations:
        for item in correlations:
            print(
                f"  {item['first']} / {item['second']}: "
                f"{item['correlation']:+.3f}"
            )
    else:
        print("  none above |0.8|")
    print(f"Wrote {arguments.output.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
