"""Fit the observed GJ 411 STIS spectrum with NUTS and dynesty."""

from run_stellar_recovery import main

from lyapy.inference.common import GJ411_CONFIGURATION, SpectrumData
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
    )
