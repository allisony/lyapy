"""Run a real E140M joint Ly-alpha and C II example.

This optional example uses one packaged Tom Ayres reduction (GJ 729 by
default), keeps both line regions on their native detector grids, and fits them
in one NUTS posterior. H I and C II share only the interstellar cloud velocity.
Their stellar velocities, columns, and Doppler widths remain independent. The
historical target-specific airglow interval is masked because these coadded IDL
products do not retain the per-exposure ``V_HELIO`` values needed by the
explicit airglow model.

Run a short mechanics check from the repository root::

    python tools/run_e140m_multiline_example.py --quick --no-progress

The quick run is not suitable for scientific interpretation. Omit ``--quick``
for the configured posterior run and inspect convergence diagnostics before
using its parameters.
"""

import argparse
import json
from pathlib import Path
import sys

try:
    import jax
    import numpyro  # noqa: F401 -- provide an actionable dependency error
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing inference dependency {error.name!r}. "
        "From the repository root, run:\n\n"
        "    python -m pip install -e '.[nuts]'"
    ) from error

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lyapy.inference import (  # noqa: E402
    LyaCiiModelBundle,
    LyaCiiModelDefaults,
    augment_configuration_with_affine_baseline,
    build_e140m_observation_operator,
    build_lya_cii_model_bundle,
    prepare_e140m_lya_cii_data,
)
from lyapy.observations import (  # noqa: E402
    E140M_TARGETS,
    GJ729_E140M_TARGET,
)


def _positive_integrated_flux(data):
    return float(
        np.trapezoid(
            np.maximum(data.flux, 0.0),
            data.wavelength,
        )
    )


def _initial_log_flux(data, fallback):
    flux = _positive_integrated_flux(data)
    return fallback if flux <= 0.0 else float(np.log10(flux))


def build_problem(target, oversample):
    """Assemble data, physical line models, and fitted local backgrounds."""
    lya_data, cii_data = prepare_e140m_lya_cii_data(target)
    lya_operator = build_e140m_observation_operator(
        lya_data.wavelength,
        aperture=target.aperture,
        lsf_profile_column=target.lsf_profile_column,
        oversample=oversample,
        model_padding=2.0,
    )
    cii_operator = build_e140m_observation_operator(
        cii_data.wavelength,
        aperture=target.aperture,
        lsf_profile_column=target.lsf_profile_column,
        oversample=oversample,
        model_padding=2.0,
    )

    # These are starting values and broad priors, not inferred facts. Users
    # should adjust them for their target and perform prior-predictive checks.
    defaults = LyaCiiModelDefaults(
        stellar_velocity=target.stellar_radial_velocity,
        stellar_velocity_bounds=(
            target.stellar_radial_velocity - 25.0,
            target.stellar_radial_velocity + 25.0,
        ),
        cii_stellar_velocity=target.stellar_radial_velocity,
        cii_stellar_velocity_bounds=(
            target.stellar_radial_velocity - 25.0,
            target.stellar_radial_velocity + 25.0,
        ),
        lya_log_total_flux=_initial_log_flux(lya_data, -11.75),
        cii_log_total_flux=_initial_log_flux(cii_data, -13.5),
    )
    bundle = build_lya_cii_model_bundle(
        lya_operator,
        cii_operator,
        defaults,
    )

    # A local affine detector/extraction residual is fitted separately in each
    # window. For a detected stellar continuum, change baseline_kind to
    # "stellar_continuum" so the baseline is ISM-absorbed and LSF-convolved.
    configuration, lya_model = augment_configuration_with_affine_baseline(
        bundle.configuration,
        bundle.lya_model,
        flux_density_scale=float(np.median(lya_data.error)),
        prefix="lya",
        baseline_kind="background",
        wavelength_scale=float(np.ptp(lya_data.wavelength) / 2.0),
        intercept_bounds=(-10.0, 10.0),
        slope_bounds=(-10.0, 10.0),
    )
    configuration, cii_model = augment_configuration_with_affine_baseline(
        configuration,
        bundle.cii_model,
        flux_density_scale=float(np.median(cii_data.error)),
        prefix="cii",
        baseline_kind="background",
        wavelength_scale=float(np.ptp(cii_data.wavelength) / 2.0),
        intercept_bounds=(-10.0, 10.0),
        slope_bounds=(-10.0, 10.0),
    )
    fitted_bundle = LyaCiiModelBundle(
        configuration,
        lya_model,
        cii_model,
    )
    return fitted_bundle.problem(
        lya_data,
        cii_data,
        name=f"{target.name} E140M Ly-alpha plus C II",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=tuple(E140M_TARGETS),
        default=GJ729_E140M_TARGET.name,
    )
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=729)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("user_data/validation/e140m-multiline"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    target = E140M_TARGETS[arguments.target]
    problem = build_problem(target, arguments.oversample)
    warmup = 20 if arguments.quick else arguments.warmup
    samples = 20 if arguments.quick else arguments.samples
    result = problem.run(
        "nuts",
        random_seed=arguments.seed,
        progress=not arguments.no_progress,
        num_warmup=warmup,
        num_samples=samples,
        reconstruct=False,
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    slug = target.name.lower().replace(" ", "")
    prefix = arguments.output_directory / f"{slug}_lya_cii"
    result_json, result_arrays = result.save(prefix.with_name(f"{prefix.name}_result"))
    figure = result.plot_all(random_seed=arguments.seed + 1)
    plot_path = prefix.with_suffix(".png")
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report = {
        "target": target.as_dict(),
        "quick_mechanics_check": arguments.quick,
        "settings": {
            "oversample": arguments.oversample,
            "warmup": warmup,
            "samples": samples,
            "seed": arguments.seed,
        },
        "shared_parameters": ["ism_velocity"],
        "line_specific_stellar_velocities": [
            "stellar_velocity",
            "cii_stellar_velocity",
        ],
        "independent_ism_parameters": {
            "H I": ["h1_log_column", "h1_doppler_b"],
            "C II": [
                "cii_log_column",
                "cii_star_log_column",
                "cii_doppler_b",
            ],
        },
        "parameter_summary": result.parameter_summary(),
        "sampler_diagnostics": result.diagnostics,
        "fit_diagnostics": {
            name: value.as_dict()
            for name, value in result.fit_diagnostics_all(
                random_seed=arguments.seed + 1
            ).items()
        },
        "files": {
            "plot": str(plot_path.resolve()),
            "result_metadata": str(result_json.resolve()),
            "result_arrays": str(result_arrays.resolve()),
        },
    }
    report_path = prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {plot_path.resolve()}")
    print(f"Wrote {report_path.resolve()}")
    if arguments.quick:
        print("Quick mode checks mechanics only; do not interpret its posterior.")


if __name__ == "__main__":
    main()
