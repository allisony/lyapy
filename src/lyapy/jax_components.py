"""Composable differentiable intrinsic emission and reversal components."""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from lyapy.jax_models import (
    KM_SPEED_OF_LIGHT,
    hydrogen_deuterium_transmission_jax,
    normalized_voigt_profile_jax,
)


class IntrinsicSpectrumResult(NamedTuple):
    """Evaluated component shapes and total-flux-normalized intrinsic spectrum."""

    intrinsic: object
    total_flux: object
    component_shapes: object
    component_weights: object
    unreversed_shape: object
    reversal_transmission: object
    normalization_integral: object


class AbsorptionSpectrumResult(NamedTuple):
    """Combined and per-component ISM transmission on one wavelength grid."""

    transmission: object
    component_transmissions: object


def _validate_index(index, name):
    if index is not None and index < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class GaussianEmissionComponent:
    """Unit-integral Gaussian emission component parameterized in velocity."""

    rest_wavelength: float
    velocity_index: int
    fwhm_index: int
    log_weight_index: int | None = None
    fixed_weight: float = 1.0

    def __post_init__(self):
        if self.rest_wavelength <= 0.0:
            raise ValueError("rest_wavelength must be positive")
        for index, name in (
            (self.velocity_index, "velocity_index"),
            (self.fwhm_index, "fwhm_index"),
            (self.log_weight_index, "log_weight_index"),
        ):
            _validate_index(index, name)
        if self.fixed_weight <= 0.0:
            raise ValueError("fixed_weight must be positive")

    def unit_profile(self, wavelength, parameters):
        center = self.rest_wavelength * (
            1.0 + parameters[self.velocity_index] / KM_SPEED_OF_LIGHT
        )
        fwhm = (
            self.rest_wavelength
            * parameters[self.fwhm_index]
            / KM_SPEED_OF_LIGHT
        )
        sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
        return jnp.exp(-0.5 * ((wavelength - center) / sigma) ** 2) / (
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )

    def weight(self, parameters):
        if self.log_weight_index is None:
            return jnp.asarray(self.fixed_weight)
        return 10.0 ** parameters[self.log_weight_index]


@dataclass(frozen=True)
class VoigtEmissionComponent:
    """Unit-integral Voigt emission component parameterized in velocity."""

    rest_wavelength: float
    velocity_index: int
    gaussian_fwhm_index: int
    lorentz_fwhm_index: int
    log_weight_index: int | None = None
    fixed_weight: float = 1.0

    def __post_init__(self):
        if self.rest_wavelength <= 0.0:
            raise ValueError("rest_wavelength must be positive")
        for index, name in (
            (self.velocity_index, "velocity_index"),
            (self.gaussian_fwhm_index, "gaussian_fwhm_index"),
            (self.lorentz_fwhm_index, "lorentz_fwhm_index"),
            (self.log_weight_index, "log_weight_index"),
        ):
            _validate_index(index, name)
        if self.fixed_weight <= 0.0:
            raise ValueError("fixed_weight must be positive")

    def unit_profile(self, wavelength, parameters):
        center = self.rest_wavelength * (
            1.0 + parameters[self.velocity_index] / KM_SPEED_OF_LIGHT
        )
        gaussian_fwhm = (
            self.rest_wavelength
            * parameters[self.gaussian_fwhm_index]
            / KM_SPEED_OF_LIGHT
        )
        lorentz_fwhm = (
            self.rest_wavelength
            * parameters[self.lorentz_fwhm_index]
            / KM_SPEED_OF_LIGHT
        )
        return normalized_voigt_profile_jax(
            wavelength,
            center,
            lorentz_fwhm,
            gaussian_fwhm,
        )

    def weight(self, parameters):
        if self.log_weight_index is None:
            return jnp.asarray(self.fixed_weight)
        return 10.0 ** parameters[self.log_weight_index]


@dataclass(frozen=True)
class GaussianOpticalDepthReversal:
    """Central reversal represented by a Gaussian optical-depth layer."""

    rest_wavelength: float
    velocity_index: int
    fwhm_index: int
    peak_optical_depth_index: int

    def __post_init__(self):
        if self.rest_wavelength <= 0.0:
            raise ValueError("rest_wavelength must be positive")
        for index, name in (
            (self.velocity_index, "velocity_index"),
            (self.fwhm_index, "fwhm_index"),
            (self.peak_optical_depth_index, "peak_optical_depth_index"),
        ):
            _validate_index(index, name)

    def transmission(self, wavelength, emission, parameters):
        del emission
        center = self.rest_wavelength * (
            1.0 + parameters[self.velocity_index] / KM_SPEED_OF_LIGHT
        )
        fwhm = (
            self.rest_wavelength
            * parameters[self.fwhm_index]
            / KM_SPEED_OF_LIGHT
        )
        sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
        optical_depth = parameters[self.peak_optical_depth_index] * jnp.exp(
            -0.5 * ((wavelength - center) / sigma) ** 2
        )
        return jnp.exp(-optical_depth)


@dataclass(frozen=True)
class LegacyIntensityReversal:
    """Legacy reversal whose optical depth follows normalized emission intensity."""

    depth_index: int

    def __post_init__(self):
        _validate_index(self.depth_index, "depth_index")

    def transmission(self, wavelength, emission, parameters):
        del wavelength
        return jnp.exp(
            -parameters[self.depth_index] * emission / jnp.max(emission)
        )


@dataclass(frozen=True)
class HydrogenDeuteriumAbsorber:
    """One untruncated H I plus D I interstellar absorption component."""

    log_h1_column_index: int
    h1_doppler_b_index: int
    h1_velocity_index: int
    deuterium_to_hydrogen_index: int | None = None
    fixed_deuterium_to_hydrogen: float = 1.5e-5

    def __post_init__(self):
        for index, name in (
            (self.log_h1_column_index, "log_h1_column_index"),
            (self.h1_doppler_b_index, "h1_doppler_b_index"),
            (self.h1_velocity_index, "h1_velocity_index"),
            (self.deuterium_to_hydrogen_index, "deuterium_to_hydrogen_index"),
        ):
            _validate_index(index, name)
        if self.fixed_deuterium_to_hydrogen <= 0.0:
            raise ValueError("fixed_deuterium_to_hydrogen must be positive")

    def transmission(self, wavelength, parameters):
        if self.deuterium_to_hydrogen_index is None:
            deuterium_to_hydrogen = self.fixed_deuterium_to_hydrogen
        else:
            deuterium_to_hydrogen = parameters[
                self.deuterium_to_hydrogen_index
            ]
        return hydrogen_deuterium_transmission_jax(
            wavelength,
            parameters[self.log_h1_column_index],
            parameters[self.h1_doppler_b_index],
            parameters[self.h1_velocity_index],
            deuterium_to_hydrogen,
        )


@dataclass(frozen=True)
class CompositeAbsorptionModel:
    """Multiply one or more duck-typed ISM transmission components."""

    components: tuple

    def __post_init__(self):
        if not self.components:
            raise ValueError("at least one absorption component is required")
        for component in self.components:
            if not hasattr(component, "transmission"):
                raise TypeError("absorption components must define transmission")

    def evaluate(self, wavelength, parameters):
        wavelength = jnp.asarray(wavelength)
        parameters = jnp.asarray(parameters)
        component_transmissions = jnp.stack(
            [
                component.transmission(wavelength, parameters)
                for component in self.components
            ]
        )
        return AbsorptionSpectrumResult(
            transmission=jnp.prod(component_transmissions, axis=0),
            component_transmissions=component_transmissions,
        )


@dataclass(frozen=True)
class CompositeEmissionModel:
    """Add unit-integral shapes and normalize post-reversal emission to total flux.

    Gaussian and Voigt components are analytically normalized over infinite
    wavelength. Reversal is treated as a localized removed-flux correction, so
    the sampled total flux is not defined by arbitrary numerical grid padding.
    """

    components: tuple
    log_total_flux_index: int
    reversal: object | None = None

    def __post_init__(self):
        if not self.components:
            raise ValueError("at least one emission component is required")
        _validate_index(self.log_total_flux_index, "log_total_flux_index")
        for component in self.components:
            if not hasattr(component, "unit_profile") or not hasattr(
                component, "weight"
            ):
                raise TypeError("emission components must define unit_profile and weight")
        if self.reversal is not None and not hasattr(self.reversal, "transmission"):
            raise TypeError("reversal must define transmission")

    def evaluate(self, wavelength, parameters):
        wavelength = jnp.asarray(wavelength)
        parameters = jnp.asarray(parameters)
        component_shapes = jnp.stack(
            [
                component.unit_profile(wavelength, parameters)
                for component in self.components
            ]
        )
        component_weights = jnp.stack(
            [component.weight(parameters) for component in self.components]
        )
        unreversed = jnp.sum(component_weights[:, None] * component_shapes, axis=0)
        if self.reversal is None:
            transmission = jnp.ones_like(unreversed)
            normalization_integral = jnp.sum(component_weights)
        else:
            transmission = self.reversal.transmission(
                wavelength,
                unreversed,
                parameters,
            )
            removed_flux = jnp.trapezoid(
                unreversed * (1.0 - transmission),
                wavelength,
            )
            normalization_integral = jnp.sum(component_weights) - removed_flux
        reversed_shape = unreversed * transmission
        total_flux = 10.0 ** parameters[self.log_total_flux_index]
        intrinsic = total_flux * reversed_shape / normalization_integral
        return IntrinsicSpectrumResult(
            intrinsic=intrinsic,
            total_flux=total_flux,
            component_shapes=component_shapes,
            component_weights=component_weights,
            unreversed_shape=unreversed,
            reversal_transmission=transmission,
            normalization_integral=normalization_integral,
        )
