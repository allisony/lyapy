"""Composable intrinsic source plus high-resolution observation operator."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp


class DetectorModelComponents(NamedTuple):
    """Model evaluated in detector bins."""

    observed: object
    point_source: object
    additive: object


class ComposableForwardResult(NamedTuple):
    """High-resolution physical model and detector-space prediction."""

    unconvolved: object
    convolved: DetectorModelComponents
    high_resolution_convolved: object
    model_wavelength: object
    absorption: object | None
    high_resolution_absorbed: object
    additive: object | None
    continuum: object | None
    high_resolution_source: object


@dataclass(frozen=True)
class ComposableBinnedModel:
    """Callable forward model compatible with the evidence framework."""

    emission_model: object
    observation_operator: object
    absorption_model: object | None = None
    additive_model: object | None = None
    continuum_model: object | None = None

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
        if self.additive_model is not None and not hasattr(
            self.additive_model,
            "evaluate",
        ):
            raise TypeError("additive_model must define evaluate")
        if self.continuum_model is not None and not hasattr(
            self.continuum_model,
            "evaluate",
        ):
            raise TypeError("continuum_model must define evaluate")

    def __call__(self, wavelength, parameters, convolution_kernel):
        del wavelength, convolution_kernel
        model_wavelength = jnp.asarray(self.observation_operator.wavelength)
        intrinsic = self.emission_model.evaluate(model_wavelength, parameters)
        if self.continuum_model is None:
            continuum = None
            high_resolution_source = intrinsic.intrinsic
        else:
            continuum = self.continuum_model.evaluate(
                model_wavelength,
                parameters,
            )
            high_resolution_source = intrinsic.intrinsic + continuum.total
        if self.absorption_model is None:
            absorption = None
            high_resolution_absorbed = high_resolution_source
        else:
            absorption = self.absorption_model.evaluate(
                model_wavelength,
                parameters,
            )
            high_resolution_absorbed = (
                high_resolution_source * absorption.transmission
            )
        high_resolution_convolved, point_source_flux = (
            self.observation_operator.apply(
                high_resolution_absorbed,
                parameters,
            )
        )
        if self.additive_model is None:
            additive = None
            additive_flux = jnp.zeros_like(point_source_flux)
        else:
            additive = self.additive_model.evaluate(
                model_wavelength,
                parameters,
            )
            additive_flux = self.observation_operator.bin(additive.total)
        detector_flux = point_source_flux + additive_flux
        return ComposableForwardResult(
            unconvolved=intrinsic,
            convolved=DetectorModelComponents(
                observed=detector_flux,
                point_source=point_source_flux,
                additive=additive_flux,
            ),
            high_resolution_convolved=high_resolution_convolved,
            model_wavelength=model_wavelength,
            absorption=absorption,
            high_resolution_absorbed=high_resolution_absorbed,
            additive=additive,
            continuum=continuum,
            high_resolution_source=high_resolution_source,
        )
