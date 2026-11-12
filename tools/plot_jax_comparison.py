"""Plot the NumPy baseline and JAX forward model with residuals."""

import argparse
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.jax_models import evaluate_voigt_lya_model_jax
from tests._legacy_reference import PARAMETERS, evaluate_gj411_legacy_case


def _plot_comparison(axes, wave, baseline, jax_values, label, scale, residual_scale):
    axes[0].plot(wave, baseline * scale, color="black", label="NumPy baseline")
    axes[0].plot(
        wave,
        jax_values * scale,
        color="#d6278b",
        linestyle="--",
        label="JAX",
    )
    axes[0].set_ylabel(label)
    axes[0].grid(alpha=0.2)

    peak = np.max(np.abs(baseline))
    residual = (jax_values - baseline) / peak * residual_scale
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].plot(wave, residual, color="#1f77b4")
    residual_power = int(np.log10(residual_scale))
    axes[1].set_ylabel(
        rf"$(\mathrm{{JAX}}-\mathrm{{NumPy}})/\mathrm{{peak}}\times10^{{{residual_power}}}$"
    )
    axes[1].grid(alpha=0.2)


def make_figure():
    reference = evaluate_gj411_legacy_case()
    wave = np.asarray(reference["wave"], dtype=np.float64)
    kernel = np.asarray(reference["lsf_kernel"], dtype=np.float64)
    result = evaluate_voigt_lya_model_jax(wave, PARAMETERS, kernel)

    jax_intrinsic = np.asarray(result.unconvolved.intrinsic)
    jax_attenuation = np.asarray(result.unconvolved.attenuation)
    jax_convolved = np.asarray(result.convolved.observed)

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    _plot_comparison(
        axes[0],
        wave,
        reference["intrinsic"],
        jax_intrinsic,
        r"Intrinsic flux density ($10^{-12}$ units)",
        1e12,
        1e6,
    )
    _plot_comparison(
        axes[1],
        wave,
        reference["ism_total"],
        jax_attenuation,
        "ISM transmission",
        1.0,
        1e5,
    )
    _plot_comparison(
        axes[2],
        wave,
        reference["model_convolved"],
        jax_convolved,
        r"Convolved flux density ($10^{-12}$ units)",
        1e12,
        1e5,
    )

    axes[0, 0].legend(frameon=False)
    axes[2, 0].xaxis.set_major_locator(MaxNLocator(7))
    axes[2, 1].xaxis.set_major_locator(MaxNLocator(5))
    axes[2, 0].set_xlabel(r"Wavelength ($\AA$)")
    axes[2, 1].set_xlabel(r"Wavelength ($\AA$)")
    figure.suptitle("GJ 411 forward-model parity: NumPy baseline vs. JAX")
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jax_model_comparison.png"),
        help="output image path (default: jax_model_comparison.png)",
    )
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    arguments = parser.parse_args()

    figure = make_figure()
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    print(f"Wrote {arguments.output.resolve()}")
    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
