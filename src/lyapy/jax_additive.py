"""Differentiable additive backgrounds and foreground emission."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from lyapy.jax_models import KM_SPEED_OF_LIGHT


class AdditiveSpectrumResult(NamedTuple):
    """Combined and per-component additive flux density."""

    total: object
    component_fluxes: object


def _validate_index(index, name):
    if index is not None and index < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AffineBackgroundComponent:
    """A locally affine residual background in scaled flux-density units.

    This component represents an additive extraction/background residual, not
    an astrophysical stellar continuum. Its value is

    ``flux_scale * (intercept + slope * (wavelength-reference)/scale)``.

    Set ``slope_index`` to ``None`` for a constant background.
    """

    reference_wavelength: float
    intercept_index: int
    slope_index: int | None = None
    flux_scale: float = 1.0
    wavelength_scale: float = 1.0

    def __post_init__(self):
        if self.reference_wavelength <= 0.0:
            raise ValueError("reference_wavelength must be positive")
        if self.flux_scale <= 0.0:
            raise ValueError("flux_scale must be positive")
        if self.wavelength_scale <= 0.0:
            raise ValueError("wavelength_scale must be positive")
        _validate_index(self.intercept_index, "intercept_index")
        _validate_index(self.slope_index, "slope_index")

    def flux_density(self, wavelength, parameters):
        wavelength = jnp.asarray(wavelength)
        intercept = parameters[self.intercept_index]
        if self.slope_index is None:
            slope = 0.0
        else:
            slope = parameters[self.slope_index]
        coordinate = (
            wavelength - self.reference_wavelength
        ) / self.wavelength_scale
        return self.flux_scale * (intercept + slope * coordinate)


@dataclass(frozen=True)
class AffineContinuumComponent(AffineBackgroundComponent):
    """A locally affine stellar continuum in scaled flux-density units.

    The functional form matches :class:`AffineBackgroundComponent`, but a
    composable model places this component before ISM absorption and the
    point-source LSF. Keeping the types distinct makes the physical ordering
    explicit in model specifications.
    """


@dataclass(frozen=True)
class GaussianAirglowComponent:
    """An empirical observed Gaussian for residual geocoronal emission.

    ``center_wavelength`` is normally computed from the CALSTIS ``V_HELIO``
    keyword. ``velocity_offset_index`` supplies a fitted offset around that
    prediction. ``scaled_total_flux_index`` is multiplied by ``flux_scale``
    and may be negative so already background-subtracted spectra can represent
    an over-subtracted geocoronal residual. The FWHM parameter is in km/s and
    represents the complete observed slit-filling profile; this component is
    detector-bin integrated without the point-source LSF.
    """

    center_wavelength: float
    scaled_total_flux_index: int
    fwhm_index: int
    velocity_offset_index: int | None = None
    flux_scale: float = 1.0

    def __post_init__(self):
        if self.center_wavelength <= 0.0:
            raise ValueError("center_wavelength must be positive")
        if self.flux_scale <= 0.0:
            raise ValueError("flux_scale must be positive")
        for index, name in (
            (self.scaled_total_flux_index, "scaled_total_flux_index"),
            (self.fwhm_index, "fwhm_index"),
            (self.velocity_offset_index, "velocity_offset_index"),
        ):
            _validate_index(index, name)

    def flux_density(self, wavelength, parameters):
        wavelength = jnp.asarray(wavelength)
        if self.velocity_offset_index is None:
            velocity_offset = 0.0
        else:
            velocity_offset = parameters[self.velocity_offset_index]
        center = self.center_wavelength * (
            1.0 + velocity_offset / KM_SPEED_OF_LIGHT
        )
        fwhm = (
            self.center_wavelength
            * parameters[self.fwhm_index]
            / KM_SPEED_OF_LIGHT
        )
        sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
        unit_profile = jnp.exp(-0.5 * ((wavelength - center) / sigma) ** 2) / (
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )
        total_flux = (
            self.flux_scale * parameters[self.scaled_total_flux_index]
        )
        return total_flux * unit_profile


@dataclass(frozen=True)
class CompositeAdditiveModel:
    """Add one or more detector-background or foreground components."""

    components: tuple

    def __post_init__(self):
        if not self.components:
            raise ValueError("at least one additive component is required")
        if not all(
            hasattr(component, "flux_density")
            for component in self.components
        ):
            raise TypeError("additive components must define flux_density")

    def evaluate(self, wavelength, parameters):
        wavelength = jnp.asarray(wavelength)
        parameters = jnp.asarray(parameters)
        component_fluxes = jnp.stack(
            [
                component.flux_density(wavelength, parameters)
                for component in self.components
            ]
        )
        return AdditiveSpectrumResult(
            total=jnp.sum(component_fluxes, axis=0),
            component_fluxes=component_fluxes,
        )
