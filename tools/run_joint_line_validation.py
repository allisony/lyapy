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
        "    python -m pip install -e '.[nuts]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference.common import (
    InferenceConfigurationBuilder,
    SpectrumData,
)
from lyapy.inference.joint import (
    JointForwardModel,
    JointSpectrumData,
)
from lyapy.inference.workflow import InferenceProblem
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


def build_joint_forward_model(
    lya_wavelength,
    si3_wavelength,
    configuration,
    oversample=8,
):
    index = configuration.parameter_names.index
    shared_velocity = index("shared_velocity")
    lya = ComposableBinnedModel(
        gaussian_emission_model_for_line_group(
            "lya",
            velocity_index=shared_velocity,
            fwhm_index=index("lya_fwhm"),
            log_total_flux_index=index("lya_log_total_flux"),
        ),
        gaussian_operator(lya_wavelength, oversample=oversample),
    )
    si3 = ComposableBinnedModel(
        gaussian_emission_model_for_line_group(
            "si_iii_1206",
            velocity_index=shared_velocity,
            fwhm_index=index("si3_fwhm"),
            log_total_flux_index=index("si3_log_total_flux"),
        ),
        gaussian_operator(si3_wavelength, oversample=oversample),
    )
    return JointForwardModel.from_mapping(
        {
            "Ly alpha": lya,
            "Si III": si3,
        }
    )


def make_configuration():
    builder = InferenceConfigurationBuilder()
    builder.add_uniform(PARAMETER_NAMES[0], TRUTH_PARAMETERS[0], -60.0, 20.0)
    builder.add_uniform(PARAMETER_NAMES[1], TRUTH_PARAMETERS[1], -12.5, -11.0)
    builder.add_uniform(PARAMETER_NAMES[2], TRUTH_PARAMETERS[2], 20.0, 160.0)
    builder.add_uniform(PARAMETER_NAMES[3], TRUTH_PARAMETERS[3], -13.8, -12.3)
    builder.add_uniform(PARAMETER_NAMES[4], TRUTH_PARAMETERS[4], 15.0, 100.0)
    return builder.build()


def make_synthetic_problem(seed, oversample=8):
    lya_wavelength = np.linspace(1215.05, 1216.28, 90)
    si3_wavelength = np.linspace(1206.20, 1206.80, 65)
    configuration = make_configuration()
    forward_model = build_joint_forward_model(
        lya_wavelength,
        si3_wavelength,
        configuration,
        oversample=oversample,
    )
    truth = jnp.asarray(configuration.base_parameters)
    empty_data = JointSpectrumData.from_mapping(
        {
            "Ly alpha": SpectrumData(
                lya_wavelength,
                np.zeros_like(lya_wavelength),
                np.ones_like(lya_wavelength),
                np.array([1.0]),
            ),
            "Si III": SpectrumData(
                si3_wavelength,
                np.zeros_like(si3_wavelength),
                np.ones_like(si3_wavelength),
                np.array([1.0]),
            ),
        }
    )
    truth_fluxes = tuple(
        np.asarray(value)
        for value in forward_model.predict(empty_data, truth)
    )
    random = np.random.default_rng(seed)
    spectra = {}
    for named, truth_flux in zip(empty_data.spectra, truth_fluxes):
        error = np.full_like(truth_flux, np.max(truth_flux) / 35.0)
        noisy_flux = truth_flux + random.normal(scale=error)
        spectra[named.name] = SpectrumData(
            named.data.wavelength,
            noisy_flux,
            error,
            named.data.convolution_kernel,
        )
    truth_by_name = dict(zip(PARAMETER_NAMES, TRUTH_PARAMETERS))
    return (
        JointSpectrumData.from_mapping(spectra),
        configuration,
        forward_model,
        truth_by_name,
        truth_fluxes,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short mechanics check; not a converged recovery fit",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("joint_line_validation"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    data, configuration, forward_model, truth, _ = (
        make_synthetic_problem(arguments.seed, arguments.oversample)
    )
    problem = InferenceProblem(
        data,
        configuration,
        forward_model,
        name="synthetic Ly-alpha and Si III joint fit",
    )
    warmup = 16 if arguments.quick else arguments.warmup
    samples = 16 if arguments.quick else arguments.samples
    result = problem.run(
        "nuts",
        random_seed=arguments.seed + 1,
        progress=not arguments.no_progress,
        num_warmup=warmup,
        num_samples=samples,
    )
    summary = result.parameter_summary()
    for name, parameter_truth in truth.items():
        summary[name]["truth"] = parameter_truth
        lower, upper = summary[name]["credible_interval"]
        summary[name]["truth_in_interval"] = (
            lower <= parameter_truth <= upper
        )
    report = {
        "problem": "synthetic Ly-alpha and Si III joint-line validation",
        "purpose": "mechanics validation; not a recommended physical Si III model",
        "seed": arguments.seed,
        "truth": truth,
        "settings": {
            "oversample": arguments.oversample,
            "warmup": warmup,
            "samples": samples,
        },
        "posterior": summary,
        "fit_diagnostics": {
            name: diagnostic.as_dict()
            for name, diagnostic in result.fit_diagnostics_all(
                random_seed=arguments.seed + 2
            ).items()
        },
        "integrated_flux": result.integrated_flux_summaries(),
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
    prefix = arguments.output_prefix
    plot_path = prefix.with_suffix(".png")
    report_path = prefix.with_suffix(".json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result_json_path, result_arrays_path = result.save(
        prefix.with_name(f"{prefix.name}_result")
    )
    figure = result.plot_all(random_seed=arguments.seed + 2)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report["files"] = {
        "plot": str(plot_path),
        "result_metadata": str(result_json_path),
        "result_arrays": str(result_arrays_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()
