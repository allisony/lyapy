"""Reviewed GJ 411 problem using the composable high-resolution model."""

from dataclasses import dataclass
from importlib import resources

import numpy as np

from lyapy.inference.common import (
    InferenceConfiguration,
    SpectrumData,
    UniformParameter,
)
from lyapy.inference.workflow import InferenceProblem
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    HydrogenDeuteriumAbsorber,
    LegacyIntensityReversal,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.observations import (
    STIS_E140M_LSF_RESOURCE,
    load_gj411_observation,
    load_tabulated_lsf,
)


COMPOSABLE_GJ411_PARAMETER_NAMES = (
    "stellar_velocity",
    "log_total_flux",
    "lorentz_fwhm",
    "gaussian_fwhm",
    "h1_column",
    "h1_doppler_b",
    "h1_velocity",
    "reversal_depth",
)

# The total flux matches the legacy GJ 411 baseline profile integrated over a
# sufficiently wide wavelength interval. Future inference fits this quantity
# directly rather than deriving it from an amplitude after sampling.
COMPOSABLE_GJ411_BASE_PARAMETERS = (
    -87.0,
    float(np.log10(1.7185359006885912e-12)),
    7.76,
    88.68,
    17.84,
    11.84,
    4.18,
    1.5,
)

GJ411_COMPOSABLE_CONFIGURATION = InferenceConfiguration(
    base_parameters=COMPOSABLE_GJ411_BASE_PARAMETERS,
    free_parameters=(
        UniformParameter("stellar_velocity", 0, -100.0, -75.0),
        UniformParameter("log_total_flux", 1, -13.0, -10.5),
        UniformParameter("lorentz_fwhm", 2, 1.0, 1000.0),
        UniformParameter("gaussian_fwhm", 3, 1.0, 1000.0),
        UniformParameter("h1_column", 4, 17.5, 18.9),
        UniformParameter("h1_doppler_b", 5, 8.6, 20.0),
        UniformParameter("h1_velocity", 6, -30.0, 30.0),
        UniformParameter("reversal_depth", 7, 0.0, 1.8),
    ),
    parameter_names=COMPOSABLE_GJ411_PARAMETER_NAMES,
)


@dataclass(frozen=True)
class ComposableInferenceProblem(InferenceProblem):
    """High-level inference problem with a composable forward model."""

    data: SpectrumData
    configuration: InferenceConfiguration
    forward_model: ComposableBinnedModel


def build_gj411_composable_forward_model(
    wavelength,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Build the reviewed stellar, ISM, and STIS operator for one grid."""
    operator = build_gj411_observation_operator(
        wavelength,
        oversample=oversample,
        model_padding=model_padding,
    )
    emission = CompositeEmissionModel(
        components=(VoigtEmissionComponent(1215.67, 0, 3, 2),),
        log_total_flux_index=1,
        reversal=LegacyIntensityReversal(7),
    )
    absorption = CompositeAbsorptionModel(
        components=(HydrogenDeuteriumAbsorber(4, 5, 6),),
    )
    return ComposableBinnedModel(
        emission,
        operator,
        absorption_model=absorption,
    )


def build_gj411_observation_operator(
    wavelength,
    *,
    oversample=8,
    model_padding=1.0,
):
    """Build the shared E140M high-resolution operator for GJ 411."""
    package = resources.files("lyapy")
    with resources.as_file(package.joinpath(STIS_E140M_LSF_RESOURCE)) as path:
        lsf = load_tabulated_lsf(path)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    dispersion = float(np.median(np.diff(wavelength)))
    operator = build_high_resolution_operator(
        wavelength,
        lsf.offset * dispersion,
        lsf.profile,
        oversample=oversample,
        model_padding=model_padding,
    )
    return operator


def load_gj411_composable_problem(
    *,
    configuration=GJ411_COMPOSABLE_CONFIGURATION,
    oversample=8,
    model_padding=1.0,
):
    """Load observed GJ 411 data and its matched composable forward model."""
    observation = load_gj411_observation()
    data = SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
    )
    forward_model = build_gj411_composable_forward_model(
        data.wavelength,
        oversample=oversample,
        model_padding=model_padding,
    )
    return ComposableInferenceProblem(
        data,
        configuration,
        forward_model,
        name="GJ 411 composable",
    )
