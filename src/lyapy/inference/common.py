"""Shared data and parameter definitions for inference backends."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm


PARAMETER_NAMES = (
    "stellar_velocity",
    "log_amplitude",
    "lorentz_fwhm",
    "gaussian_fwhm",
    "h1_column",
    "h1_doppler_b",
    "h1_velocity",
    "h1_column_2",
    "h1_doppler_b_2",
    "h1_velocity_2",
    "deuterium_to_hydrogen",
    "reversal_depth",
)


@dataclass(frozen=True)
class UniformParameter:
    """One free parameter with a bounded uniform prior."""

    name: str
    index: int
    lower: float
    upper: float

    def __post_init__(self):
        if not 0 <= self.index < len(PARAMETER_NAMES):
            raise ValueError(
                f"parameter index {self.index} is outside the model vector"
            )
        if self.name != PARAMETER_NAMES[self.index]:
            raise ValueError(
                f"parameter {self.name!r} does not match index {self.index}"
            )
        if self.lower >= self.upper:
            raise ValueError(
                f"invalid bounds for {self.name}: {self.lower}, {self.upper}"
            )

    def transform_unit_interval(self, value):
        """Map a unit-interval value to this prior."""
        return self.lower + value * (self.upper - self.lower)


@dataclass(frozen=True)
class TruncatedNormalParameter:
    """One free parameter with a bounded normal prior."""

    name: str
    index: int
    lower: float
    upper: float
    mean: float
    standard_deviation: float

    def __post_init__(self):
        UniformParameter(self.name, self.index, self.lower, self.upper)
        if not self.lower <= self.mean <= self.upper:
            raise ValueError(
                f"mean for {self.name} must lie within its bounds"
            )
        if self.standard_deviation <= 0.0:
            raise ValueError(
                f"standard deviation for {self.name} must be positive"
            )

    def transform_unit_interval(self, value):
        """Map a unit-interval value through the truncated-normal quantile."""
        lower_standardized = (
            self.lower - self.mean
        ) / self.standard_deviation
        upper_standardized = (
            self.upper - self.mean
        ) / self.standard_deviation
        return truncnorm.ppf(
            value,
            lower_standardized,
            upper_standardized,
            loc=self.mean,
            scale=self.standard_deviation,
        )


@dataclass(frozen=True)
class InferenceConfiguration:
    """Fixed model vector plus the subset varied by an inference backend."""

    base_parameters: tuple[float, ...]
    free_parameters: tuple[UniformParameter | TruncatedNormalParameter, ...]

    def __post_init__(self):
        if len(self.base_parameters) != len(PARAMETER_NAMES):
            raise ValueError(
                f"base_parameters must contain {len(PARAMETER_NAMES)} values"
            )
        if not self.free_parameters:
            raise ValueError("at least one free parameter is required")
        indices = [parameter.index for parameter in self.free_parameters]
        names = [parameter.name for parameter in self.free_parameters]
        if len(indices) != len(set(indices)) or len(names) != len(set(names)):
            raise ValueError("free parameter names and indices must be unique")

    @property
    def ndim(self):
        return len(self.free_parameters)

    def assemble_numpy(self, free_values):
        free_values = np.asarray(free_values, dtype=np.float64)
        if free_values.shape != (self.ndim,):
            raise ValueError(
                f"expected {self.ndim} free values, got {free_values.shape}"
            )
        full_parameters = np.asarray(self.base_parameters, dtype=np.float64).copy()
        full_parameters[[parameter.index for parameter in self.free_parameters]] = (
            free_values
        )
        return full_parameters

    def assemble_jax(self, free_values):
        full_parameters = jnp.asarray(self.base_parameters, dtype=jnp.float64)
        indices = jnp.asarray(
            [parameter.index for parameter in self.free_parameters], dtype=jnp.int32
        )
        return full_parameters.at[indices].set(jnp.asarray(free_values))

    def transform_unit_cube(self, unit_values):
        unit_values = np.asarray(unit_values, dtype=np.float64)
        if unit_values.shape != (self.ndim,):
            raise ValueError(
                f"expected {self.ndim} unit values, got {unit_values.shape}"
            )
        return np.asarray(
            [
                parameter.transform_unit_interval(value)
                for parameter, value in zip(self.free_parameters, unit_values)
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SpectrumData:
    """One spectrum and its convolution kernel for likelihood evaluation."""

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    convolution_kernel: np.ndarray

    def __post_init__(self):
        arrays = {
            "wavelength": np.asarray(self.wavelength, dtype=np.float64),
            "flux": np.asarray(self.flux, dtype=np.float64),
            "error": np.asarray(self.error, dtype=np.float64),
            "convolution_kernel": np.asarray(
                self.convolution_kernel, dtype=np.float64
            ),
        }
        if arrays["wavelength"].ndim != 1 or arrays["wavelength"].size < 2:
            raise ValueError("wavelength must be a one-dimensional array")
        for name in ("flux", "error"):
            if arrays[name].shape != arrays["wavelength"].shape:
                raise ValueError(f"{name} must match the wavelength shape")
        if (
            arrays["convolution_kernel"].ndim != 1
            or arrays["convolution_kernel"].size == 0
        ):
            raise ValueError("convolution_kernel must be one-dimensional")
        if not all(np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError("spectrum arrays must contain only finite values")
        if np.any(arrays["error"] <= 0.0):
            raise ValueError("error values must be positive")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


GJ411_CONFIGURATION = InferenceConfiguration(
    base_parameters=(
        -87.0,
        -10.06,
        7.76,
        88.68,
        17.84,
        11.84,
        4.18,
        0.0,
        11.5,
        5.87,
        1.5e-5,
        1.5,
    ),
    free_parameters=(
        UniformParameter("stellar_velocity", 0, -100.0, -75.0),
        UniformParameter("log_amplitude", 1, -18.0, -8.0),
        UniformParameter("lorentz_fwhm", 2, 1.0, 1000.0),
        UniformParameter("gaussian_fwhm", 3, 1.0, 1000.0),
        UniformParameter("h1_column", 4, 17.5, 18.9),
        UniformParameter("h1_doppler_b", 5, 8.6, 20.0),
        UniformParameter("h1_velocity", 6, -30.0, 30.0),
        UniformParameter("reversal_depth", 11, 1.0, 1.8),
    ),
)
