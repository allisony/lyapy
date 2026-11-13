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
import matplotlib.pyplot as plt

from lyapy.inference import (
    InferenceConfiguration,
    InferenceProblem,
    InferenceResult,
    PriorSensitivityScenario,
    SpectrumData,
    UniformParameter,
    load_prior_sensitivity,
    run_prior_sensitivity,
    sample_prior,
    with_prior_bounds,
)


class Unconvolved(NamedTuple):
    intrinsic: object
    total_flux: object


class Convolved(NamedTuple):
    observed: object


class Result(NamedTuple):
    model_wavelength: object
    unconvolved: Unconvolved
    convolved: Convolved


@dataclass(frozen=True)
class GaussianModel:
    width: float

    def __call__(self, wavelength, parameters, convolution_kernel):
        del convolution_kernel
        profile = jnp.exp(-0.5 * (wavelength / self.width) ** 2)
        profile = profile / jnp.trapezoid(profile, wavelength)
        intrinsic = parameters[0] * profile
        return Result(
            wavelength,
            Unconvolved(intrinsic, parameters[0]),
            Convolved(intrinsic),
        )


def make_configuration(lower=0.5, upper=1.5):
    return InferenceConfiguration(
        base_parameters=(1.0,),
        free_parameters=(
            UniformParameter("total_flux", 0, lower, upper),
        ),
        parameter_names=("total_flux",),
    )


@pytest.fixture(scope="module")
def base_problem():
    wavelength = np.linspace(-1.5, 1.5, 17)
    truth_model = GaussianModel(0.35)
    configuration = make_configuration()
    truth = np.asarray(
        truth_model(wavelength, jnp.asarray((1.0,)), np.ones(1))
        .convolved.observed
    )
    data = SpectrumData(
        wavelength,
        truth,
        np.full_like(truth, 0.04),
        np.ones(1),
    )
    return InferenceProblem(
        data,
        configuration,
        truth_model,
        name="narrow Gaussian",
    )


def test_prior_sampling_and_prediction_are_reproducible(base_problem):
    first = base_problem.prior_predictive(draws=32, random_seed=71)
    repeated = base_problem.prior_predictive(draws=32, random_seed=71)

    np.testing.assert_allclose(first.samples, repeated.samples)
    np.testing.assert_allclose(
        first.predictions["spectrum"],
        repeated.predictions["spectrum"],
    )
    assert first.samples.shape == (32, 1)
    assert np.all((first.samples >= 0.5) & (first.samples <= 1.5))
    assert first.prediction_bands().shape == (5, 17)
    assert first.integrated_flux_summary()["mean"] > 0.0
    assert first.as_dict()["draws"] == 32

    figure = first.plot(maximum_curves=5)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_explicit_prior_bound_replacement_preserves_configuration(base_problem):
    narrowed = with_prior_bounds(
        base_problem.configuration,
        {"total_flux": (0.8, 1.2)},
    )
    draws = sample_prior(narrowed, draws=50, random_seed=73)

    assert narrowed.base_parameters == base_problem.configuration.base_parameters
    assert np.all((draws >= 0.8) & (draws <= 1.2))
    with pytest.raises(KeyError, match="unknown"):
        with_prior_bounds(narrowed, {"missing": (0.0, 1.0)})
    with pytest.raises(ValueError, match="outside new bounds"):
        with_prior_bounds(narrowed, {"total_flux": (1.1, 1.2)})


def test_boundary_diagnostics_flag_posterior_pressure(base_problem):
    samples = np.array([[1.46], [1.47], [1.48], [1.49]])
    predictions = np.vstack(
        [
            np.asarray(
                base_problem.forward_model(
                    base_problem.data.wavelength,
                    jnp.asarray(value),
                    base_problem.data.convolution_kernel,
                ).convolved.observed
            )
            for value in samples
        ]
    )
    result = InferenceResult(
        name=base_problem.name,
        backend="nuts",
        parameter_names=("total_flux",),
        samples=samples,
        weights=None,
        diagnostics={},
        problem_specification=base_problem.as_dict(),
        observations={"spectrum": base_problem.data},
        predictions={"spectrum": predictions},
    )

    diagnostic = result.prior_boundary_diagnostics()[0]

    assert diagnostic.pressure == "upper"
    assert diagnostic.upper_boundary_probability == 1.0
    assert diagnostic.lower_boundary_probability == 0.0
    json.dumps(diagnostic.as_dict())


@pytest.fixture(scope="module")
def sensitivity_result(base_problem):
    wide_problem = InferenceProblem(
        base_problem.data,
        base_problem.configuration,
        GaussianModel(0.7),
        name="wide Gaussian",
    )
    narrow_configuration = with_prior_bounds(
        base_problem.configuration,
        {"total_flux": (0.8, 1.2)},
    )
    scenarios = (
        PriorSensitivityScenario(
            "broad flux prior",
            (base_problem, wide_problem),
            description="Uniform total flux from 0.5 to 1.5",
        ),
        PriorSensitivityScenario(
            "narrow flux prior",
            (
                replace(base_problem, configuration=narrow_configuration),
                replace(wide_problem, configuration=narrow_configuration),
            ),
            description="Uniform total flux from 0.8 to 1.2",
        ),
    )
    return run_prior_sensitivity(
        scenarios,
        random_seed=79,
        progress=False,
        nlive=20,
        dlogz=10.0,
        sample="rwalk",
        bootstrap=0,
    )


def test_prior_sensitivity_reports_rankings_and_pairwise_stability(
    sensitivity_result,
):
    result = sensitivity_result
    pairwise = result.pairwise_summary("narrow Gaussian", "wide Gaussian")

    assert len(result.scenarios) == 2
    assert result.model_names == ("narrow Gaussian", "wide Gaussian")
    assert isinstance(result.winner_stable, bool)
    assert isinstance(result.ranking_stable, bool)
    assert len(pairwise["scenarios"]) == 2
    assert pairwise["delta_log_evidence_span"] >= 0.0
    json.dumps(result.as_dict())

    figure = result.plot("narrow Gaussian", "wide Gaussian")
    assert len(figure.axes) == 1
    plt.close(figure)


def test_prior_sensitivity_saves_every_scenarios_priors(
    sensitivity_result,
    tmp_path,
):
    manifest = sensitivity_result.save(tmp_path / "sensitivity")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert set(metadata["scenarios"]) == {
        "broad flux prior",
        "narrow flux prior",
    }
    narrow_model = metadata["scenarios"]["narrow flux prior"][
        "comparison"
    ]["models"]["narrow Gaussian"]
    free_parameter = narrow_model["result"]["problem"]["hypothesis"][
        "free_parameters"
    ][0]
    assert free_parameter["lower"] == 0.8
    assert free_parameter["upper"] == 1.2
    assert (
        tmp_path
        / "sensitivity"
        / metadata["scenarios"]["narrow flux prior"]["comparison_file"]
    ).exists()
    loaded = load_prior_sensitivity(tmp_path / "sensitivity")
    assert loaded.as_dict() == sensitivity_result.as_dict()


def test_prior_sensitivity_rejects_model_changes_disguised_as_prior_changes(
    base_problem,
):
    wide_problem = InferenceProblem(
        base_problem.data,
        base_problem.configuration,
        GaussianModel(0.7),
        name="wide Gaussian",
    )
    changed_model = InferenceProblem(
        base_problem.data,
        base_problem.configuration,
        GaussianModel(0.4),
        name=base_problem.name,
    )
    scenarios = (
        PriorSensitivityScenario("reference", (base_problem, wide_problem)),
        PriorSensitivityScenario("changed", (changed_model, wide_problem)),
    )

    with pytest.raises(ValueError, match="not model structure"):
        run_prior_sensitivity(scenarios, progress=False)
