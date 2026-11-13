"""A fully worked, commented lyapy inference example.

This script deliberately constructs the GJ 411 Ly-alpha problem one piece at
a time. It is meant to be copied and edited for another star or another model.
The convenience function ``load_gj411_composable_problem`` constructs the same
kind of object with fewer lines, but would hide the choices this tutorial is
intended to explain.

Install the optional inference dependencies and run a short mechanics check:

    python -m pip install -e '.[inference]'
    python tools/example_inference_workflow.py --quick --no-progress

The ``--quick`` result is intentionally too short for scientific use. Omit it
for the documented starting settings, then assess convergence and prior
sensitivity before interpreting the posterior.
"""

import argparse
from importlib import resources
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

# Lyapy's scientific comparisons use double precision. This must be enabled
# before importing and evaluating JAX-backed model code.
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


# Allow this file to run directly from a source checkout without installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference import (  # noqa: E402
    InferenceConfigurationBuilder,
    InferenceProblem,
    SpectrumData,
    load_inference_result,
)
from lyapy.jax_components import (  # noqa: E402
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    HydrogenDeuteriumAbsorber,
    LegacyIntensityReversal,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel  # noqa: E402
from lyapy.jax_instrument import build_high_resolution_operator  # noqa: E402
from lyapy.likelihoods import gaussian_log_likelihood  # noqa: E402
from lyapy.observations import (  # noqa: E402
    STIS_E140M_LSF_RESOURCE,
    load_gj411_observation,
    load_tabulated_lsf,
)


def make_data():
    """Return the observed arrays in the backend-neutral data container."""
    # This helper reproduces the reviewed preprocessing for the packaged
    # spectrum: fit a linear continuum in fixed off-line windows, subtract it,
    # crop to the Ly-alpha fit window, and prepare the tabulated E140M LSF.
    # For another target, users would replace this call with their own
    # wavelength, continuum-subtracted flux, uncertainty, and LSF arrays.
    observation = load_gj411_observation()

    # SpectrumData uses four one-dimensional arrays:
    #   wavelength: detector-pixel centers in Angstrom
    #   flux:       continuum-subtracted flux density
    #   error:      positive 1-sigma uncertainty for each flux value
    #   kernel:     a normalized detector-grid convolution kernel
    #
    # The high-resolution composable model below owns its more accurate LSF
    # operator. The kernel is still retained here because SpectrumData also
    # supports simpler forward models that convolve directly on the data grid.
    data = SpectrumData(
        wavelength=observation.wavelength,
        flux=observation.flux,
        error=observation.error,
        convolution_kernel=observation.convolution_kernel,
    )
    return data


def make_configuration():
    """Define the parameter vector, initial values, and prior distributions."""
    # Components access parameters by integer index because indexed arrays are
    # efficient inside compiled JAX functions. The builder assigns those
    # indices and checks that names, initial values, and priors stay aligned.
    parameters = InferenceConfigurationBuilder()

    # Every add_uniform call returns the new parameter's integer index. Units
    # are noted below. These ranges reproduce the current GJ 411 starting
    # hypothesis; they are not automatic or universal astrophysical priors.
    stellar_velocity = parameters.add_uniform(
        "stellar_velocity",
        initial=-87.0,  # km/s
        lower=-100.0,
        upper=-75.0,
    )
    log_total_flux = parameters.add_uniform(
        "log_total_flux",
        initial=float(np.log10(1.7185359006885912e-12)),
        lower=-13.0,
        upper=-10.5,
    )
    lorentz_fwhm = parameters.add_uniform(
        "lorentz_fwhm",
        initial=7.76,  # km/s
        lower=1.0,
        upper=1000.0,
    )
    gaussian_fwhm = parameters.add_uniform(
        "gaussian_fwhm",
        initial=88.68,  # km/s
        lower=1.0,
        upper=1000.0,
    )
    h1_column = parameters.add_uniform(
        "h1_column",
        initial=17.84,  # log10 column density in cm^-2
        lower=17.5,
        upper=18.9,
    )
    h1_doppler_b = parameters.add_uniform(
        "h1_doppler_b",
        initial=11.84,  # km/s
        lower=8.6,
        upper=20.0,
    )
    h1_velocity = parameters.add_uniform(
        "h1_velocity",
        initial=4.18,  # km/s in the same frame as the spectrum
        lower=-30.0,
        upper=30.0,
    )
    reversal_depth = parameters.add_uniform(
        "reversal_depth",
        initial=1.5,  # dimensionless legacy reversal strength
        lower=0.0,
        upper=1.8,
    )

    # build() freezes the initial full parameter vector and the subset that is
    # free. Use add_fixed(...) instead of add_uniform(...) for any quantity
    # that should remain fixed. Truncated-normal priors are also available via
    # add_truncated_normal(...).
    configuration = parameters.build()
    indices = {
        "stellar_velocity": stellar_velocity,
        "log_total_flux": log_total_flux,
        "lorentz_fwhm": lorentz_fwhm,
        "gaussian_fwhm": gaussian_fwhm,
        "h1_column": h1_column,
        "h1_doppler_b": h1_doppler_b,
        "h1_velocity": h1_velocity,
        "reversal_depth": reversal_depth,
    }
    return configuration, indices


def make_forward_model(data, indices, oversample=8, model_padding=1.0):
    """Assemble emission, ISM absorption, and the STIS observation operator."""
    # The packaged STIS table tabulates the LSF against detector-pixel offset.
    # The high-resolution operator needs wavelength offsets, so multiply by the
    # measured Angstrom-per-pixel dispersion of this spectrum.
    package = resources.files("lyapy")
    with resources.as_file(package.joinpath(STIS_E140M_LSF_RESOURCE)) as path:
        tabulated_lsf = load_tabulated_lsf(path)
    detector_dispersion = float(np.median(np.diff(data.wavelength)))
    lsf_offsets_angstrom = tabulated_lsf.offset * detector_dispersion

    # oversample controls numerical resolution, not a fitted physical
    # parameter. model_padding extends the physical wavelength grid beyond the
    # fitted detector bins so absorption wings and LSF convolution are not
    # artificially cut off at the displayed window edges.
    observation_operator = build_high_resolution_operator(
        data.wavelength,
        lsf_offsets_angstrom,
        tabulated_lsf.profile,
        oversample=oversample,
        model_padding=model_padding,
    )

    # This stellar hypothesis is one Voigt emission component. Its velocity,
    # Gaussian FWHM, and Lorentzian FWHM read the corresponding entries in the
    # shared parameter vector. CompositeEmissionModel then scales the profile
    # so 10**log_total_flux is the post-reversal integrated intrinsic flux.
    emission = CompositeEmissionModel(
        components=(
            VoigtEmissionComponent(
                rest_wavelength=1215.67,
                velocity_index=indices["stellar_velocity"],
                gaussian_fwhm_index=indices["gaussian_fwhm"],
                lorentz_fwhm_index=indices["lorentz_fwhm"],
            ),
        ),
        log_total_flux_index=indices["log_total_flux"],
        reversal=LegacyIntensityReversal(indices["reversal_depth"]),
    )

    # One interstellar cloud contributes both H I and D I absorption. D/H is
    # fixed to 1.5e-5 here; pass a deuterium_to_hydrogen_index if it should be a
    # parameter. Additional clouds can be appended to this components tuple.
    absorption = CompositeAbsorptionModel(
        components=(
            HydrogenDeuteriumAbsorber(
                log_h1_column_index=indices["h1_column"],
                h1_doppler_b_index=indices["h1_doppler_b"],
                h1_velocity_index=indices["h1_velocity"],
                fixed_deuterium_to_hydrogen=1.5e-5,
            ),
        ),
    )

    # The final callable evaluates the intrinsic emission on the oversampled
    # grid, applies ISM transmission, convolves with the LSF, and integrates
    # the result over each detector-pixel bin in the physically correct order.
    return ComposableBinnedModel(
        emission,
        observation_operator,
        absorption_model=absorption,
    )


def make_problem(oversample=8, model_padding=1.0):
    """Bundle all scientific choices required to define one hypothesis."""
    data = make_data()
    configuration, indices = make_configuration()
    forward_model = make_forward_model(
        data,
        indices,
        oversample=oversample,
        model_padding=model_padding,
    )

    # gaussian_log_likelihood assumes independent normally distributed errors
    # with the supplied 1-sigma values. Alternative noise models can be passed
    # here without changing the inference workflow.
    return InferenceProblem(
        data=data,
        configuration=configuration,
        forward_model=forward_model,
        likelihood_function=gaussian_log_likelihood,
        name="GJ 411 single Voigt with legacy self-reversal",
    )


def run_problem(problem, backend, random_seed, quick, progress):
    """Run the selected sampler while keeping backend choices in one place."""
    if backend == "nuts":
        # NUTS is the usual choice for posterior parameter estimation. Warmup
        # adapts the sampler; samples retained after warmup represent the
        # posterior. NUTS does not calculate Bayesian evidence.
        return problem.run(
            "nuts",
            random_seed=random_seed,
            progress=progress,
            num_warmup=25 if quick else 500,
            num_samples=25 if quick else 1000,
            target_accept_probability=0.9,
        )

    # Dynesty is useful when the evidence is needed for comparing hypotheses.
    # nlive controls the number of live points; dlogz is the evidence-based
    # stopping tolerance. Its posterior draws carry unequal weights, which the
    # InferenceResult summary methods apply automatically.
    return problem.run(
        "dynesty",
        random_seed=random_seed,
        progress=progress,
        nlive=20 if quick else 500,
        dlogz=10.0 if quick else 0.05,
        sample="rwalk",
        bootstrap=0,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nuts", "dynesty"), default="nuts")
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--model-padding", type=float, default=1.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("inference-example-output"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a very short mechanics check, not a scientific fit",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    problem = make_problem(
        oversample=arguments.oversample,
        model_padding=arguments.model_padding,
    )
    result = run_problem(
        problem,
        backend=arguments.backend,
        random_seed=arguments.random_seed,
        quick=arguments.quick,
        progress=not arguments.no_progress,
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = arguments.output_directory / f"gj411_{arguments.backend}_fit"
    json_path, arrays_path = result.save(output_stem)
    # Loading requires only the JSON path; it points to the adjacent NPZ file.
    # This round trip demonstrates that analysis and plotting do not require a
    # live sampler object or unsafe Python-object pickling.
    restored_result = load_inference_result(json_path)
    figure_path = output_stem.with_suffix(".png")
    restored_result.plot(random_seed=arguments.random_seed).savefig(
        figure_path,
        dpi=180,
    )
    plt.close("all")

    print("Parameter summary:")
    print(json.dumps(restored_result.parameter_summary(), indent=2))
    print("Fit diagnostics:")
    print(
        json.dumps(
            restored_result.fit_diagnostics(
                random_seed=arguments.random_seed
            ).as_dict(),
            indent=2,
        )
    )
    print(f"Saved metadata: {json_path}")
    print(f"Saved numerical arrays: {arrays_path}")
    print(f"Saved fit plot: {figure_path}")
    if result.log_evidence is not None:
        print(
            "log evidence: "
            f"{result.log_evidence:.3f} +/- {result.log_evidence_error:.3f}"
        )
    if arguments.quick:
        print("Quick mode checks mechanics only; do not interpret this posterior.")


if __name__ == "__main__":
    main()
