"""Validate a shared-velocity joint fit to synthetic Ly-alpha and Si III."""

import argparse
import json
from pathlib import Path
import sys

try:
    import jax
    import numpyro  # noqa: F401 -- fail early with an actionable message
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. "
        "From the repository root, run:\n\n"
        "    python -m pip install -e '.[inference]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.diagnostics import summarize_nuts
from lyapy.inference.joint import (
    JointForwardModel,
    JointSpectrumData,
    NamedForwardModel,
    NamedSpectrumData,
)
from lyapy.inference.numpyro_backend import run_nuts
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.spectral_lines import (
    SPECTRAL_LINE_GROUPS,
    gaussian_emission_model_for_line_group,
)


PARAMETER_NAMES = (
    "shared_velocity",
    "lya_log_total_flux",
    "lya_fwhm",
    "si3_log_total_flux",
    "si3_fwhm",
)
TRUTH_PARAMETERS = (-20.0, -11.8, 80.0, -13.0, 45.0)


def gaussian_operator(wavelength, sigma=0.018, oversample=8):
    offsets = np.linspace(-0.12, 0.12, 241)
    profile = np.exp(-0.5 * (offsets / sigma) ** 2)
    return build_high_resolution_operator(
        wavelength,
        offsets,
        profile,
        oversample=oversample,
        model_padding=0.3,
    )


def build_joint_forward_model(lya_wavelength, si3_wavelength, oversample=8):
    lya = ComposableBinnedModel(
        gaussian_emission_model_for_line_group(
            "lya",
            velocity_index=0,
            fwhm_index=2,
            log_total_flux_index=1,
        ),
        gaussian_operator(lya_wavelength, oversample=oversample),
    )
    si3 = ComposableBinnedModel(
        gaussian_emission_model_for_line_group(
            "si_iii_1206",
            velocity_index=0,
            fwhm_index=4,
            log_total_flux_index=3,
        ),
        gaussian_operator(si3_wavelength, oversample=oversample),
    )
    return JointForwardModel(
        (
            NamedForwardModel("Ly alpha", lya),
            NamedForwardModel("Si III", si3),
        )
    )


def make_configuration():
    return InferenceConfiguration(
        base_parameters=TRUTH_PARAMETERS,
        free_parameters=(
            UniformParameter(PARAMETER_NAMES[0], 0, -60.0, 20.0),
            UniformParameter(PARAMETER_NAMES[1], 1, -12.5, -11.0),
            UniformParameter(PARAMETER_NAMES[2], 2, 20.0, 160.0),
            UniformParameter(PARAMETER_NAMES[3], 3, -13.8, -12.3),
            UniformParameter(PARAMETER_NAMES[4], 4, 15.0, 100.0),
        ),
        parameter_names=PARAMETER_NAMES,
    )


def make_synthetic_problem(seed, oversample=8):
    lya_wavelength = np.linspace(1215.05, 1216.28, 90)
    si3_wavelength = np.linspace(1206.20, 1206.80, 65)
    forward_model = build_joint_forward_model(
        lya_wavelength,
        si3_wavelength,
        oversample=oversample,
    )
    configuration = make_configuration()
    truth = jnp.asarray(configuration.base_parameters)
    empty_data = JointSpectrumData(
        (
            NamedSpectrumData(
                "Ly alpha",
                SpectrumData(
                    lya_wavelength,
                    np.zeros_like(lya_wavelength),
                    np.ones_like(lya_wavelength),
                    np.array([1.0]),
                ),
            ),
            NamedSpectrumData(
                "Si III",
                SpectrumData(
                    si3_wavelength,
                    np.zeros_like(si3_wavelength),
                    np.ones_like(si3_wavelength),
                    np.array([1.0]),
                ),
            ),
        )
    )
    truth_fluxes = tuple(
        np.asarray(value)
        for value in forward_model.predict(empty_data, truth)
    )
    random = np.random.default_rng(seed)
    spectra = []
    for named, truth_flux in zip(empty_data.spectra, truth_fluxes):
        error = np.full_like(truth_flux, np.max(truth_flux) / 35.0)
        noisy_flux = truth_flux + random.normal(scale=error)
        spectra.append(
            NamedSpectrumData(
                named.name,
                SpectrumData(
                    named.data.wavelength,
                    noisy_flux,
                    error,
                    named.data.convolution_kernel,
                ),
            )
        )
    truth_by_name = dict(zip(PARAMETER_NAMES, TRUTH_PARAMETERS))
    return (
        JointSpectrumData(tuple(spectra)),
        configuration,
        forward_model,
        truth_by_name,
        truth_fluxes,
    )


def make_plot(data, samples, truth_fluxes):
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    for axis, named, truth_flux in zip(axes, data.spectra, truth_fluxes):
        site_name = named.site_name
        predictions = np.asarray(samples[f"model_flux_{site_name}"])
        lower, median, upper = np.percentile(predictions, (5.0, 50.0, 95.0), axis=0)
        scale = 10.0 ** np.floor(np.log10(np.max(truth_flux)))
        axis.errorbar(
            named.data.wavelength,
            named.data.flux / scale,
            yerr=named.data.error / scale,
            color="0.55",
            linewidth=0.8,
            label="synthetic data",
        )
        axis.fill_between(
            named.data.wavelength,
            lower / scale,
            upper / scale,
            color="tab:blue",
            alpha=0.22,
            label="90% posterior band",
        )
        axis.plot(
            named.data.wavelength,
            median / scale,
            color="tab:blue",
            label="posterior median",
        )
        axis.plot(
            named.data.wavelength,
            truth_flux / scale,
            color="black",
            linestyle="--",
            label="truth",
        )
        axis.set_title(named.name)
        axis.set_ylabel(f"flux density / {scale:.0e}")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel(r"wavelength ($\AA$)")
    figure.suptitle("Synthetic joint-line recovery with shared stellar velocity")
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("joint_line_validation"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    data, configuration, forward_model, truth, truth_fluxes = (
        make_synthetic_problem(arguments.seed, arguments.oversample)
    )
    sampler = run_nuts(
        data,
        configuration,
        jax.random.key(arguments.seed + 1),
        num_warmup=arguments.warmup,
        num_samples=arguments.samples,
        progress_bar=not arguments.no_progress,
        forward_model=forward_model,
    )
    samples = sampler.get_samples()
    summary = summarize_nuts(
        sampler,
        PARAMETER_NAMES,
        truth=truth,
    )
    report = {
        "problem": "synthetic Ly-alpha and Si III joint-line validation",
        "purpose": "mechanics validation; not a recommended physical Si III model",
        "seed": arguments.seed,
        "truth": truth,
        "settings": {
            "oversample": arguments.oversample,
            "warmup": arguments.warmup,
            "samples": arguments.samples,
        },
        "posterior": summary,
        "shared_parameters": ["shared_velocity"],
        "line_specific_parameters": {
            "Ly alpha": ["lya_log_total_flux", "lya_fwhm"],
            "Si III": ["si3_log_total_flux", "si3_fwhm"],
        },
        "spectral_line_groups": {
            key: SPECTRAL_LINE_GROUPS[key].as_dict()
            for key in ("lya", "si_iii_1206")
        },
    }
    figure = make_plot(data, samples, truth_fluxes)
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
