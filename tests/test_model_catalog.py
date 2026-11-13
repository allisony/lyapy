import json

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.inference.composable import build_gj411_observation_operator
from lyapy.inference.model_catalog import (
    build_gj411_model_catalog,
    make_model_catalog_entry,
)
from lyapy.jax_components import CompositeEmissionModel, GaussianEmissionComponent


@pytest.fixture(scope="module")
def catalog():
    wavelength = np.linspace(1214.5, 1216.55, 160)
    return build_gj411_model_catalog(
        wavelength,
        oversample=4,
        model_padding=0.5,
    )


def test_catalog_contains_curated_noncombinatorial_model_set(catalog):
    assert tuple(catalog) == (
        "single_gaussian",
        "single_voigt",
        "double_gaussian",
        "gaussian_gaussian_reversal",
        "voigt_legacy_reversal",
        "voigt_gaussian_reversal",
    )
    assert len(catalog) == 6
    assert len({entry.name for entry in catalog.values()}) == len(catalog)


@pytest.mark.parametrize(
    "name",
    (
        "single_gaussian",
        "single_voigt",
        "double_gaussian",
        "gaussian_gaussian_reversal",
        "voigt_legacy_reversal",
        "voigt_gaussian_reversal",
    ),
)
def test_catalog_model_predicts_detector_spectrum(catalog, name):
    entry = catalog[name]
    wavelength = np.linspace(1214.5, 1216.55, 160)
    result = entry.forward_model(
        wavelength,
        jnp.asarray(entry.configuration.base_parameters),
        np.array([1.0]),
    )

    assert result.convolved.observed.shape == wavelength.shape
    assert np.all(np.isfinite(result.convolved.observed))
    assert np.all(result.convolved.observed >= 0.0)
    assert tuple(
        parameter.name for parameter in entry.configuration.free_parameters
    ) == entry.configuration.parameter_names


def test_double_gaussian_uses_ordered_width_priors(catalog):
    configuration = catalog["double_gaussian"].configuration
    priors = {
        parameter.name: parameter
        for parameter in configuration.free_parameters
    }

    assert priors["narrow_fwhm"].upper <= priors["broad_fwhm"].lower
    assert "ordered-widths" in catalog["double_gaussian"].tags


def test_reversal_prescriptions_are_explicit_and_serializable(catalog):
    legacy = catalog["voigt_legacy_reversal"]
    gaussian = catalog["voigt_gaussian_reversal"]

    assert "reversal_depth" in legacy.configuration.parameter_names
    assert "reversal_peak_optical_depth" in gaussian.configuration.parameter_names
    assert "reversal_fwhm" in gaussian.configuration.parameter_names
    assert legacy.components != gaussian.components
    json.dumps(legacy.as_dict())
    json.dumps(gaussian.as_dict())


def test_user_defined_components_can_be_registered_as_catalog_entry():
    wavelength = np.linspace(1214.5, 1216.55, 160)
    operator = build_gj411_observation_operator(
        wavelength,
        oversample=4,
        model_padding=0.5,
    )
    configuration = InferenceConfiguration(
        base_parameters=(-80.0, -11.8, 100.0),
        free_parameters=(
            UniformParameter("velocity", 0, -150.0, 50.0),
            UniformParameter("log_total_flux", 1, -13.0, -10.0),
            UniformParameter("fwhm", 2, 5.0, 500.0),
        ),
        parameter_names=("velocity", "log_total_flux", "fwhm"),
    )
    emission = CompositeEmissionModel(
        (GaussianEmissionComponent(1215.67, 0, 2),),
        1,
    )
    entry = make_model_catalog_entry(
        "custom",
        "User-provided Gaussian example",
        configuration,
        emission,
        operator,
        components=("Custom Gaussian",),
        tags=("user-defined",),
    )

    result = entry.forward_model(
        wavelength,
        jnp.asarray(configuration.base_parameters),
        np.array([1.0]),
    )
    assert result.convolved.observed.shape == wavelength.shape
    assert entry.hypothesis().forward_model is entry.forward_model
    assert entry.as_dict()["tags"] == ["user-defined"]
