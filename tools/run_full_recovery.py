"""Run full eight-parameter GJ 411 recovery with NUTS and dynesty."""

from run_stellar_recovery import main

from lyapy.inference.common import GJ411_CONFIGURATION


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


if __name__ == "__main__":
    main(
        configuration=GJ411_CONFIGURATION,
        parameter_labels=PARAMETER_LABELS,
        problem="synthetic GJ 411 full eight-parameter recovery",
        plot_title="Eight-parameter synthetic GJ 411 recovery",
        default_output_prefix="full_recovery",
        description=__doc__,
    )
