"""Fit and reconstruct the observed GJ 411 STIS spectrum."""

import matplotlib.pyplot as plt
import numpy as np

from run_stellar_recovery import main, normalized_dynesty_weights

from lyapy.inference.common import GJ411_CONFIGURATION, SpectrumData
from lyapy.inference.reconstruction import (
    DEFAULT_PERCENTILE_PROBABILITIES,
    evaluate_reconstruction_samples,
    posterior_percentile_bands,
    summarize_integrated_flux,
)
from lyapy.inference.diagnostics import weighted_quantile
from lyapy.observations import load_gj411_observation


PARAMETER_LABELS = (
    "stellar velocity",
    "log amplitude",
    "Lorentz FWHM",
    "Gaussian FWHM",
    "H I column",
    "H I Doppler b",
    "H I velocity",
    "reversal depth",
)


def make_gj411_problem(seed, configuration=GJ411_CONFIGURATION):
    """Load the reviewed GJ 411 observation for inference."""
    del seed
    observation = load_gj411_observation()
    data = SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
    )
    return data, configuration, None


def make_reconstruction_figure(
    data,
    nuts_reconstruction,
    dynesty_reconstruction,
    dynesty_weights,
    probabilities,
):
    """Plot profile uncertainty bands and integrated-flux posteriors."""
    nuts_intrinsic = posterior_percentile_bands(
        nuts_reconstruction.intrinsic,
        probabilities,
    )
    nuts_observed = posterior_percentile_bands(
        nuts_reconstruction.convolved_observed,
        probabilities,
    )
    dynesty_intrinsic = posterior_percentile_bands(
        dynesty_reconstruction.intrinsic,
        probabilities,
        dynesty_weights,
    )
    dynesty_observed = posterior_percentile_bands(
        dynesty_reconstruction.convolved_observed,
        probabilities,
        dynesty_weights,
    )

    figure, (profile_axis, flux_axis) = plt.subplots(
        2,
        1,
        figsize=(9.5, 8.5),
        gridspec_kw={"height_ratios": (3, 1)},
    )
    wave = data.wavelength
    scale = 1.0e12
    profile_axis.errorbar(
        wave,
        data.flux * scale,
        yerr=data.error * scale,
        color="0.65",
        linewidth=0.8,
        label="GJ 411 STIS data",
    )
    profile_axis.fill_between(
        wave,
        nuts_intrinsic[0] * scale,
        nuts_intrinsic[4] * scale,
        color="#ff7f0e",
        alpha=0.12,
        label="intrinsic 95% interval",
    )
    profile_axis.fill_between(
        wave,
        nuts_intrinsic[1] * scale,
        nuts_intrinsic[3] * scale,
        color="#ff7f0e",
        alpha=0.28,
        label="intrinsic 68% interval",
    )
    profile_axis.plot(
        wave,
        nuts_intrinsic[2] * scale,
        color="#ff7f0e",
        linewidth=2.0,
        label="NUTS intrinsic median",
    )
    profile_axis.fill_between(
        wave,
        nuts_observed[1] * scale,
        nuts_observed[3] * scale,
        color="#d6278b",
        alpha=0.25,
        label="fitted 68% interval",
    )
    profile_axis.plot(
        wave,
        nuts_observed[2] * scale,
        color="#d6278b",
        linewidth=2.0,
        label="NUTS fitted median",
    )
    profile_axis.plot(
        wave,
        dynesty_intrinsic[2] * scale,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="dynesty intrinsic median",
    )
    profile_axis.plot(
        wave,
        dynesty_observed[2] * scale,
        color="#17becf",
        linestyle="--",
        linewidth=1.5,
        label="dynesty fitted median",
    )
    profile_axis.set_ylabel(r"flux density ($10^{-12}$ units)")
    profile_axis.legend(frameon=False, ncol=2, fontsize="small")
    profile_axis.grid(alpha=0.2)

    nuts_flux = nuts_reconstruction.integrated_intrinsic_flux * scale
    dynesty_flux = dynesty_reconstruction.integrated_intrinsic_flux * scale
    nuts_limits = np.quantile(nuts_flux, (0.005, 0.995))
    dynesty_limits = weighted_quantile(
        dynesty_flux,
        (0.005, 0.995),
        dynesty_weights,
    )
    lower = min(nuts_limits[0], dynesty_limits[0])
    upper = max(nuts_limits[1], dynesty_limits[1])
    padding = 0.08 * (upper - lower)
    histogram_range = (lower - padding, upper + padding)
    flux_axis.hist(
        nuts_flux,
        bins=45,
        range=histogram_range,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#d6278b",
        label="NumPyro/NUTS",
    )
    flux_axis.hist(
        dynesty_flux,
        bins=45,
        range=histogram_range,
        weights=dynesty_weights,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="#1f77b4",
        label="dynesty",
    )
    flux_axis.set_xlabel(
        r"integrated intrinsic Ly$\alpha$ flux ($10^{-12}$ erg s$^{-1}$ cm$^{-2}$)"
    )
    flux_axis.set_ylabel("posterior density")
    flux_axis.legend(frameon=False)
    flux_axis.grid(alpha=0.2)
    figure.suptitle("GJ 411 posterior Ly-alpha reconstruction")
    figure.tight_layout()
    return figure, {
        "nuts_intrinsic": nuts_intrinsic,
        "nuts_convolved_observed": nuts_observed,
        "dynesty_intrinsic": dynesty_intrinsic,
        "dynesty_convolved_observed": dynesty_observed,
    }


def write_reconstruction_products(
    data,
    configuration,
    nuts,
    nested,
    parameter_names,
    prefix,
    seed,
):
    """Create reusable posterior reconstruction arrays, summaries, and plot."""
    del seed
    nuts_samples = nuts.get_samples()
    nuts_values = np.column_stack(
        [np.asarray(nuts_samples[name]) for name in parameter_names]
    )
    dynesty_values = np.asarray(nested.results.samples)
    dynesty_weights = normalized_dynesty_weights(nested)
    nuts_reconstruction = evaluate_reconstruction_samples(
        data,
        configuration,
        nuts_values,
    )
    dynesty_reconstruction = evaluate_reconstruction_samples(
        data,
        configuration,
        dynesty_values,
    )
    probabilities = np.asarray(DEFAULT_PERCENTILE_PROBABILITIES)
    figure, bands = make_reconstruction_figure(
        data,
        nuts_reconstruction,
        dynesty_reconstruction,
        dynesty_weights,
        probabilities,
    )
    figure_path = prefix.with_name(prefix.name + "_reconstruction.png")
    arrays_path = prefix.with_name(prefix.name + "_reconstruction.npz")
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    np.savez_compressed(
        arrays_path,
        wavelength=data.wavelength,
        percentile_probabilities=probabilities,
        nuts_integrated_intrinsic_flux=(
            nuts_reconstruction.integrated_intrinsic_flux
        ),
        dynesty_integrated_intrinsic_flux=(
            dynesty_reconstruction.integrated_intrinsic_flux
        ),
        dynesty_weights=dynesty_weights,
        **bands,
    )
    report = {
        "reconstruction": {
            "integration_wavelength_range_angstrom": [
                float(data.wavelength[0]),
                float(data.wavelength[-1]),
            ],
            "integrated_flux_units": "erg s^-1 cm^-2",
            "nuts_integrated_intrinsic_flux": summarize_integrated_flux(
                nuts_reconstruction.integrated_intrinsic_flux,
                probabilities,
            ),
            "dynesty_integrated_intrinsic_flux": summarize_integrated_flux(
                dynesty_reconstruction.integrated_intrinsic_flux,
                probabilities,
                dynesty_weights,
            ),
        }
    }
    return report, [figure_path, arrays_path]


if __name__ == "__main__":
    main(
        configuration=GJ411_CONFIGURATION,
        parameter_labels=PARAMETER_LABELS,
        problem="observed GJ 411 STIS E140M inference",
        plot_title="Observed GJ 411 STIS E140M inference",
        default_output_prefix="gj411_inference",
        description=__doc__,
        problem_factory=make_gj411_problem,
        data_label="GJ 411 STIS data",
        postprocess=write_reconstruction_products,
    )
