from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
pytest.importorskip("numpyro")

from lyapy.inference import (
    InferenceConfiguration,
    InferenceProblem,
    SpectrumData,
    UniformParameter,
    load_inference_result,
)
from lyapy.inference.dynesty_backend import build_dynesty_functions
from lyapy.likelihoods import gaussian_log_likelihood


class Convolved(NamedTuple):
    observed: object


class ModelResult(NamedTuple):
    convolved: Convolved


@dataclass(frozen=True)
class ConstantModel:
    def __call__(self, wavelength, parameters, convolution_kernel):
        del convolution_kernel
        return ModelResult(Convolved(jnp.full(wavelength.shape, parameters[0])))


@pytest.fixture
def masked_problem():
    data = SpectrumData(
        wavelength=np.arange(4.0),
        flux=np.array((1.0, 1000.0, 1.0, 1.0)),
        error=np.ones(4),
        convolution_kernel=np.ones(1),
        likelihood_mask=np.array((True, False, True, True)),
        metadata={"reason": "synthetic airglow mask"},
    )
    configuration = InferenceConfiguration(
        base_parameters=(1.0,),
        free_parameters=(UniformParameter("level", 0, 0.0, 2.0),),
        parameter_names=("level",),
    )
    return InferenceProblem(data, configuration, ConstantModel())


def test_dynesty_likelihood_ignores_masked_detector_pixels(masked_problem):
    log_likelihood, _ = build_dynesty_functions(
        masked_problem.data,
        masked_problem.configuration,
        forward_model=masked_problem.forward_model,
    )

    expected = gaussian_log_likelihood(np.ones(3), np.ones(3), np.ones(3))
    assert log_likelihood(np.array((1.0,))) == pytest.approx(expected)


def test_nuts_runs_with_a_static_likelihood_mask(masked_problem, tmp_path):
    result = masked_problem.run(
        "nuts",
        random_seed=7,
        progress=False,
        reconstruct=False,
        num_warmup=8,
        num_samples=8,
    )

    assert result.draws == 8
    assert result.observations["spectrum"].fitted_pixels == 3
    assert result.pointwise_log_likelihoods["spectrum"].shape == (8, 3)
    diagnostics = result.fit_diagnostics(random_seed=11)
    assert diagnostics.as_dict()["pixels"] == 4
    assert diagnostics.as_dict()["fitted_pixels"] == 3
    assert diagnostics.as_dict()["excluded_pixels"] == 1
    assert 1 not in diagnostics.large_residual_indices

    json_path, _ = result.save(tmp_path / "masked-result")
    restored = load_inference_result(json_path)
    np.testing.assert_array_equal(
        restored.observations["spectrum"].likelihood_mask,
        masked_problem.data.likelihood_mask,
    )
    assert restored.observations["spectrum"].metadata == (
        masked_problem.data.metadata
    )
    np.testing.assert_allclose(
        restored.pointwise_log_likelihoods["spectrum"],
        result.pointwise_log_likelihoods["spectrum"],
    )


def test_spectrum_data_validates_and_preserves_mask_metadata():
    with pytest.raises(ValueError, match="at least one"):
        SpectrumData(
            np.arange(3.0),
            np.ones(3),
            np.ones(3),
            np.ones(1),
            likelihood_mask=np.zeros(3, dtype=bool),
        )
    with pytest.raises(ValueError, match="likelihood_mask"):
        SpectrumData(
            np.arange(3.0),
            np.ones(3),
            np.ones(3),
            np.ones(1),
            likelihood_mask=np.ones(2, dtype=bool),
        )
