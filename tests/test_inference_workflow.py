from dataclasses import dataclass, replace
import json
from typing import NamedTuple

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
pytest.importorskip("numpyro")
pytest.importorskip("dynesty")

from lyapy.inference import (
    InferenceConfiguration,
    InferenceProblem,
    JointForwardModel,
    JointSpectrumData,
    NamedForwardModel,
    NamedSpectrumData,
    SpectrumData,
    UniformParameter,
    load_inference_result,
)


class Unconvolved(NamedTuple):
    intrinsic: object
    total_flux: object


class Convolved(NamedTuple):
    observed: object


class PhysicalResult(NamedTuple):
    model_wavelength: object
    unconvolved: Unconvolved
    convolved: Convolved


@dataclass(frozen=True)
class GaussianLineModel:
    amplitude_index: int
    center: float
    width: float

    def __call__(self, wavelength, parameters, convolution_kernel):
        del convolution_kernel
        profile = jnp.exp(
            -0.5 * ((wavelength - self.center) / self.width) ** 2
        )
        profile = profile / jnp.trapezoid(profile, wavelength)
        amplitude = parameters[self.amplitude_index]
        intrinsic = amplitude * profile
        return PhysicalResult(
            wavelength,
            Unconvolved(intrinsic, amplitude),
            Convolved(intrinsic),
        )


def make_spectrum(model, wavelength, amplitude=1.0, error=0.04):
    configuration = InferenceConfiguration(
        base_parameters=(amplitude,),
        free_parameters=(UniformParameter("total_flux", 0, 0.5, 1.5),),
        parameter_names=("total_flux",),
    )
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(
        model(wavelength, jnp.asarray((amplitude,)), np.ones(1))
        .convolved.observed
    )
    data = SpectrumData(
        wavelength,
        flux,
        np.full_like(flux, error),
        np.ones(1),
    )
    return data, configuration


@pytest.fixture(scope="module")
def single_problem():
    model = GaussianLineModel(0, center=0.0, width=0.35)
    data, configuration = make_spectrum(
        model,
        np.linspace(-1.5, 1.5, 17),
    )
    return InferenceProblem(
        data,
        configuration,
        model,
        name="synthetic Gaussian",
    )


@pytest.fixture(scope="module")
def nuts_result(single_problem):
    return single_problem.run(
        "nuts",
        random_seed=11,
        progress=False,
        reconstruct=False,
        num_warmup=12,
        num_samples=16,
    )


@pytest.fixture(scope="module")
def joint_problem():
    first_model = GaussianLineModel(0, center=0.0, width=0.3)
    second_model = GaussianLineModel(0, center=5.0, width=0.5)
    first, configuration = make_spectrum(
        first_model,
        np.linspace(-1.25, 1.25, 15),
    )
    second, _ = make_spectrum(
        second_model,
        np.linspace(3.5, 6.5, 13),
        error=0.05,
    )
    data = JointSpectrumData.from_mapping(
        {"line A": first, "line B": second}
    )
    forward_model = JointForwardModel(
        (
            NamedForwardModel("line A", first_model),
            NamedForwardModel("line B", second_model),
        )
    )
    return InferenceProblem(
        data,
        configuration,
        forward_model,
        name="joint synthetic lines",
    )


@pytest.fixture(scope="module")
def dynesty_result(joint_problem):
    return joint_problem.run(
        "dynesty",
        random_seed=19,
        progress=False,
        nlive=20,
        dlogz=10.0,
        sample="rwalk",
        bootstrap=0,
    )


def test_problem_records_a_reproducible_specification(single_problem):
    specification = single_problem.as_dict()

    assert specification["name"] == "synthetic Gaussian"
    assert specification["data"]["type"] == "single"
    assert specification["data"]["spectra"]["spectrum"]["pixels"] == 17
    assert specification["hypothesis"]["free_parameters"][0]["prior"] == (
        "UniformParameter"
    )


def test_nuts_workflow_returns_backend_neutral_products(nuts_result):
    result = nuts_result

    assert result.backend == "nuts"
    assert result.samples.shape == (16, 1)
    assert result.weights is None
    assert result.log_evidence is None
    assert result.reconstructions == {}
    assert result.predictions["spectrum"].shape == (16, 17)
    assert result.parameter_samples()["total_flux"].shape == (16,)
    assert result.parameter_summary()["total_flux"]["mean"] > 0.0
    assert result.prediction_bands(probabilities=(0.5,)).shape == (1, 17)


def test_dynesty_workflow_reconstructs_each_joint_spectrum(dynesty_result):
    result = dynesty_result

    assert result.backend == "dynesty"
    assert result.weights.shape == (result.draws,)
    np.testing.assert_allclose(np.sum(result.weights), 1.0)
    assert np.isfinite(result.log_evidence)
    assert result.log_evidence_error >= 0.0
    assert set(result.predictions) == {"line A", "line B"}
    assert set(result.reconstructions) == {"line A", "line B"}
    assert result.predictions["line A"].shape == (result.draws, 15)
    assert result.predictions["line B"].shape == (result.draws, 13)
    flux = result.integrated_flux_summary("line A")
    assert 0.5 <= flux["mean"] <= 1.5


def test_result_round_trip_uses_json_and_non_pickle_arrays(
    dynesty_result,
    tmp_path,
):
    json_path, arrays_path = dynesty_result.save(tmp_path / "joint-fit")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["arrays_file"] == "joint-fit.npz"
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["posterior_samples"].dtype != object
        assert arrays["spectrum_0_intrinsic"].dtype != object

    loaded = load_inference_result(tmp_path / "joint-fit")
    assert loaded.as_dict() == dynesty_result.as_dict()
    np.testing.assert_allclose(loaded.samples, dynesty_result.samples)
    np.testing.assert_allclose(loaded.weights, dynesty_result.weights)
    for name in dynesty_result.predictions:
        np.testing.assert_allclose(
            loaded.predictions[name],
            dynesty_result.predictions[name],
        )
        np.testing.assert_allclose(
            loaded.reconstructions[name].intrinsic,
            dynesty_result.reconstructions[name].intrinsic,
        )


def test_workflow_rejects_invalid_requests_and_result_shapes(
    single_problem,
    nuts_result,
):
    with pytest.raises(ValueError, match="backend"):
        single_problem.run("emcee", progress=False)
    with pytest.raises(ValueError, match="random_seed"):
        single_problem.run("nuts", random_seed=-1, progress=False)
    with pytest.raises(ValueError, match="reconstruction_batch_size"):
        single_problem.run(
            "nuts",
            progress=False,
            reconstruction_batch_size=0,
        )
    with pytest.raises(ValueError, match="samples must match"):
        replace(nuts_result, samples=np.ones((nuts_result.draws, 2)))


def test_joint_problem_requires_a_predict_method(joint_problem):
    with pytest.raises(TypeError, match="define predict"):
        InferenceProblem(
            joint_problem.data,
            joint_problem.configuration,
            GaussianLineModel(0, center=0.0, width=0.3),
        )
