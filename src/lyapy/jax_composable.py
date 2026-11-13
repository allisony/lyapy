"""Composable intrinsic source plus high-resolution observation operator."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp


class DetectorModelComponents(NamedTuple):
    """Model evaluated in detector bins."""

    observed: object


class ComposableForwardResult(NamedTuple):
    """High-resolution physical model and detector-space prediction."""

    unconvolved: object
    convolved: DetectorModelComponents
    high_resolution_convolved: object
    model_wavelength: object
    absorption: object | None
    high_resolution_absorbed: object


@dataclass(frozen=True)
class ComposableBinnedModel:
    """Callable forward model compatible with the evidence framework."""

    emission_model: object
    observation_operator: object
    absorption_model: object | None = None

    def __post_init__(self):
        if not hasattr(self.emission_model, "evaluate"):
            raise TypeError("emission_model must define evaluate")
        if not hasattr(self.observation_operator, "apply"):
            raise TypeError("observation_operator must define apply")
        if self.absorption_model is not None and not hasattr(
            self.absorption_model,
            "evaluate",
        ):
            raise TypeError("absorption_model must define evaluate")

    def __call__(self, wavelength, parameters, convolution_kernel):
        del wavelength, convolution_kernel
        model_wavelength = jnp.asarray(self.observation_operator.wavelength)
        intrinsic = self.emission_model.evaluate(model_wavelength, parameters)
        if self.absorption_model is None:
            absorption = None
            high_resolution_absorbed = intrinsic.intrinsic
        else:
            absorption = self.absorption_model.evaluate(
                model_wavelength,
                parameters,
            )
            high_resolution_absorbed = (
                intrinsic.intrinsic * absorption.transmission
            )
        high_resolution_convolved, detector_flux = self.observation_operator.apply(
            high_resolution_absorbed,
            parameters,
        )
        return ComposableForwardResult(
            unconvolved=intrinsic,
            convolved=DetectorModelComponents(observed=detector_flux),
            high_resolution_convolved=high_resolution_convolved,
            model_wavelength=model_wavelength,
            absorption=absorption,
            high_resolution_absorbed=high_resolution_absorbed,
        )
