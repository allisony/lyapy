import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from lyapy.inference.common import (
    GJ411_CONFIGURATION,
    SpectrumData,
    TruncatedNormalParameter,
    UniformParameter,
)
from lyapy.inference.identifiability import (
    analyze_local_identifiability,
    prior_standard_deviation,
)
from lyapy.observations import load_gj411_observation


def test_prior_standard_deviation_supports_shared_prior_types():
    uniform = UniformParameter("log_amplitude", 1, -10.3, -9.8)
    normal = TruncatedNormalParameter(
        "log_amplitude",
        1,
        -10.3,
        -9.8,
        mean=-10.05,
        standard_deviation=0.1,
    )

    np.testing.assert_allclose(
        prior_standard_deviation(uniform),
        0.5 / np.sqrt(12.0),
    )
    assert 0.0 < prior_standard_deviation(normal) < normal.standard_deviation


@pytest.fixture(scope="module")
def gj411_identifiability():
    observation = load_gj411_observation()
    data = SpectrumData(
        observation.wavelength,
        observation.flux,
        observation.error,
        observation.convolution_kernel,
    )
    return analyze_local_identifiability(data, GJ411_CONFIGURATION)


def test_gj411_identifiability_shapes_and_values(gj411_identifiability):
    result = gj411_identifiability
    parameter_count = GJ411_CONFIGURATION.ndim

    assert result.jacobian.shape == (159, parameter_count)
    assert result.singular_values.shape == (parameter_count,)
    assert result.posterior_correlation.shape == (
        parameter_count,
        parameter_count,
    )
    assert np.all(np.isfinite(result.jacobian))
    assert np.all(result.singular_values >= 0.0)
    assert 0 < result.numerical_rank <= parameter_count
    np.testing.assert_allclose(np.diag(result.posterior_correlation), 1.0)
    assert np.all((result.posterior_to_prior_scale > 0.0))
    assert np.all((result.posterior_to_prior_scale <= 1.0))


def test_identifiability_report_is_compact_and_serializable(gj411_identifiability):
    report = gj411_identifiability.as_dict()

    assert report["parameter_names"] == list(
        parameter.name for parameter in GJ411_CONFIGURATION.free_parameters
    )
    assert len(report["weakest_direction"]) == GJ411_CONFIGURATION.ndim
    assert "jacobian" not in report
    assert "local linear" in report["caveat"]
