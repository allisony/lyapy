"""Joint inference over multiple named spectra with shared parameters."""

from dataclasses import dataclass
import re

import jax.numpy as jnp
import numpy as np

from lyapy.inference.common import SpectrumData
from lyapy.inference.noise import (
    evaluate_jax_log_likelihood,
    evaluate_numpy_log_likelihood,
)


def _callable_name(function):
    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


def _site_name(name):
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not normalized:
        raise ValueError(f"spectrum name {name!r} cannot form a sample-site name")
    return normalized


@dataclass(frozen=True)
class NamedSpectrumData:
    """One uniquely named spectrum in a joint inference problem."""

    name: str
    data: SpectrumData

    def __post_init__(self):
        if not self.name:
            raise ValueError("spectrum name must not be empty")
        if not isinstance(self.data, SpectrumData):
            raise TypeError("data must be SpectrumData")

    @property
    def site_name(self):
        """Return a NumPyro-safe form of the spectrum name."""
        return _site_name(self.name)


@dataclass(frozen=True)
class JointSpectrumData:
    """Two or more named spectra evaluated in one posterior."""

    spectra: tuple[NamedSpectrumData, ...]

    def __post_init__(self):
        spectra = tuple(self.spectra)
        if len(spectra) < 2:
            raise ValueError("joint inference requires at least two spectra")
        if not all(isinstance(spectrum, NamedSpectrumData) for spectrum in spectra):
            raise TypeError("spectra must contain NamedSpectrumData values")
        names = [spectrum.name for spectrum in spectra]
        site_names = [spectrum.site_name for spectrum in spectra]
        if len(names) != len(set(names)):
            raise ValueError("joint spectrum names must be unique")
        if len(site_names) != len(set(site_names)):
            raise ValueError("joint spectrum names must have unique site names")
        object.__setattr__(self, "spectra", spectra)

    @property
    def names(self):
        return tuple(spectrum.name for spectrum in self.spectra)

    @classmethod
    def from_mapping(cls, spectra):
        """Build joint data while preserving mapping insertion order."""
        return cls(
            tuple(
                NamedSpectrumData(name, data)
                for name, data in spectra.items()
            )
        )


def _validated_coadd_spectra(spectra):
    spectra = tuple(spectra)
    if len(spectra) < 2:
        raise ValueError("coaddition requires at least two spectra")
    if not all(isinstance(spectrum, SpectrumData) for spectrum in spectra):
        raise TypeError("spectra must contain SpectrumData values")
    reference = spectra[0]
    for spectrum in spectra[1:]:
        if not np.array_equal(spectrum.wavelength, reference.wavelength):
            raise ValueError(
                "inverse-variance coaddition requires identical wavelength "
                "grids; retain the native grids and use a joint fit"
            )
        if not np.array_equal(
            spectrum.convolution_kernel,
            reference.convolution_kernel,
        ):
            raise ValueError(
                "inverse-variance coaddition requires identical convolution "
                "kernels; use a joint fit for exposure-specific LSFs"
            )
        if not np.array_equal(
            spectrum.likelihood_mask,
            reference.likelihood_mask,
        ):
            raise ValueError(
                "inverse-variance coaddition requires identical likelihood "
                "masks; use a joint fit for exposure-specific masking"
            )
    return spectra


def inverse_variance_coadd(spectra):
    """Coadd spectra only in the exact Gaussian-equivalence case.

    The inputs must share a wavelength grid, convolution kernel, and likelihood
    mask. Under independent Gaussian errors and one common detector-space
    prediction, the returned spectrum contains exactly the same
    parameter-dependent likelihood as the joint exposures. No interpolation or
    covariance approximation is performed.
    """
    spectra = _validated_coadd_spectra(spectra)
    reference = spectra[0]
    inverse_variances = np.stack(
        [1.0 / spectrum.error**2 for spectrum in spectra]
    )
    total_inverse_variance = np.sum(inverse_variances, axis=0)
    flux = np.sum(
        inverse_variances
        * np.stack([spectrum.flux for spectrum in spectra]),
        axis=0,
    ) / total_inverse_variance
    error = 1.0 / np.sqrt(total_inverse_variance)
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "coadd_method": "inverse_variance",
            "coadd_exposure_count": len(spectra),
            "coadd_likelihood_equivalence": (
                "independent Gaussian errors, identical wavelength grids, "
                "identical detector-space predictions"
            ),
            "coadd_source_metadata": [
                dict(spectrum.metadata) for spectrum in spectra
            ],
        }
    )
    return SpectrumData(
        reference.wavelength,
        flux,
        error,
        reference.convolution_kernel,
        likelihood_mask=reference.likelihood_mask,
        metadata=metadata,
    )


def gaussian_coadd_log_likelihood_offset(spectra):
    """Return the model-independent joint-minus-coadd Gaussian constant."""
    spectra = _validated_coadd_spectra(spectra)
    coadded = inverse_variance_coadd(spectra)
    mask = coadded.likelihood_mask
    flux = np.stack([spectrum.flux[mask] for spectrum in spectra])
    error = np.stack([spectrum.error[mask] for spectrum in spectra])
    inverse_variance = 1.0 / error**2
    coadded_flux = coadded.flux[mask]
    scatter = np.sum(inverse_variance * (flux - coadded_flux) ** 2)
    joint_normalization = np.sum(np.log(2.0 * np.pi * error**2))
    coadd_normalization = np.sum(
        np.log(2.0 * np.pi * coadded.error[mask] ** 2)
    )
    return float(
        -0.5 * (scatter + joint_normalization - coadd_normalization)
    )


@dataclass(frozen=True)
class NamedForwardModel:
    """One named forward model corresponding to one named spectrum."""

    name: str
    forward_model: object

    def __post_init__(self):
        if not self.name:
            raise ValueError("forward-model name must not be empty")
        if not callable(self.forward_model):
            raise TypeError("forward_model must be callable")


@dataclass(frozen=True)
class JointForwardModel:
    """Line-specific forward models evaluated with one parameter vector."""

    models: tuple[NamedForwardModel, ...]

    def __post_init__(self):
        models = tuple(self.models)
        if len(models) < 2:
            raise ValueError("joint inference requires at least two forward models")
        if not all(isinstance(model, NamedForwardModel) for model in models):
            raise TypeError("models must contain NamedForwardModel values")
        names = [model.name for model in models]
        if len(names) != len(set(names)):
            raise ValueError("joint forward-model names must be unique")
        object.__setattr__(self, "models", models)

    @property
    def names(self):
        return tuple(model.name for model in self.models)

    @classmethod
    def from_mapping(cls, models):
        """Build named forward models while preserving mapping order."""
        return cls(
            tuple(
                NamedForwardModel(name, forward_model)
                for name, forward_model in models.items()
            )
        )

    def predict(self, data, parameters):
        """Return detector-space predictions in joint-data order."""
        if not isinstance(data, JointSpectrumData):
            raise TypeError("data must be JointSpectrumData")
        models = {model.name: model.forward_model for model in self.models}
        if set(models) != set(data.names):
            raise ValueError("joint data and forward-model names must match")
        return tuple(
            models[spectrum.name](
                spectrum.data.wavelength,
                parameters,
                spectrum.data.convolution_kernel,
            ).convolved.observed
            for spectrum in data.spectra
        )

    def __call__(self, data, parameters):
        return self.predict(data, parameters)

    def as_dict(self):
        return {
            "type": "joint",
            "models": {
                model.name: _callable_name(model.forward_model)
                for model in self.models
            },
        }


@dataclass(frozen=True)
class NamedLikelihood:
    """One likelihood function corresponding to one named spectrum."""

    name: str
    likelihood_function: object

    def __post_init__(self):
        if not self.name:
            raise ValueError("likelihood name must not be empty")
        if not callable(self.likelihood_function):
            raise TypeError("likelihood_function must be callable")


@dataclass(frozen=True)
class JointLikelihood:
    """Optional line-specific likelihood functions for a joint fit."""

    likelihoods: tuple[NamedLikelihood, ...]

    def __post_init__(self):
        likelihoods = tuple(self.likelihoods)
        if len(likelihoods) < 2:
            raise ValueError("joint inference requires at least two likelihoods")
        if not all(
            isinstance(likelihood, NamedLikelihood)
            for likelihood in likelihoods
        ):
            raise TypeError("likelihoods must contain NamedLikelihood values")
        names = [likelihood.name for likelihood in likelihoods]
        if len(names) != len(set(names)):
            raise ValueError("joint likelihood names must be unique")
        object.__setattr__(self, "likelihoods", likelihoods)

    def likelihood_for(self, name):
        for likelihood in self.likelihoods:
            if likelihood.name == name:
                return likelihood.likelihood_function
        raise KeyError(name)

    @classmethod
    def from_mapping(cls, likelihoods):
        """Build named likelihoods while preserving mapping order."""
        return cls(
            tuple(
                NamedLikelihood(name, likelihood_function)
                for name, likelihood_function in likelihoods.items()
            )
        )

    def __call__(self, observed, predicted, error, parameters):
        raise TypeError("JointLikelihood must be evaluated with JointSpectrumData")

    def as_dict(self):
        specifications = {}
        for likelihood in self.likelihoods:
            function = likelihood.likelihood_function
            specification = {"callable": _callable_name(function)}
            if hasattr(function, "as_dict"):
                specification["settings"] = function.as_dict()
            specifications[likelihood.name] = specification
        return {"distribution": "joint", "spectra": specifications}


def _likelihood_for(likelihood_function, name):
    if isinstance(likelihood_function, JointLikelihood):
        return likelihood_function.likelihood_for(name)
    return likelihood_function


def evaluate_joint_numpy_log_likelihoods(
    likelihood_function,
    data,
    predictions,
    parameters,
):
    """Return one NumPy log-likelihood contribution per named spectrum."""
    if len(predictions) != len(data.spectra):
        raise ValueError("predictions must match the number of joint spectra")
    return tuple(
        evaluate_numpy_log_likelihood(
            _likelihood_for(likelihood_function, spectrum.name),
            spectrum.data.flux[spectrum.data.likelihood_indices],
            prediction[spectrum.data.likelihood_indices],
            spectrum.data.error[spectrum.data.likelihood_indices],
            parameters,
        )
        for spectrum, prediction in zip(data.spectra, predictions)
    )


def evaluate_joint_jax_log_likelihoods(
    likelihood_function,
    data,
    predictions,
    parameters,
):
    """Return one JAX log-likelihood contribution per named spectrum."""
    if len(predictions) != len(data.spectra):
        raise ValueError("predictions must match the number of joint spectra")
    return tuple(
        evaluate_jax_log_likelihood(
            _likelihood_for(likelihood_function, spectrum.name),
            jnp.asarray(spectrum.data.flux)[
                spectrum.data.likelihood_indices
            ],
            prediction[spectrum.data.likelihood_indices],
            jnp.asarray(spectrum.data.error)[
                spectrum.data.likelihood_indices
            ],
            parameters,
        )
        for spectrum, prediction in zip(data.spectra, predictions)
    )
