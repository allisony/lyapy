import numpy as np
import pytest

from lyapy.inference.common import (
    InferenceConfigurationBuilder,
    TruncatedNormalParameter,
    UniformParameter,
)


def test_builder_assigns_indices_and_preserves_explicit_shared_parameter():
    builder = InferenceConfigurationBuilder()
    velocity = builder.add_uniform("stellar_velocity", -20.0, -60.0, 20.0)
    lya_flux = builder.add_uniform("lya_log_total_flux", -11.8, -12.5, -11.0)
    si3_flux = builder.add_uniform("si3_log_total_flux", -13.0, -13.8, -12.3)
    fixed_ratio = builder.add_fixed("fixed_doublet_ratio", 2.0)

    assert velocity == 0
    assert lya_flux == 1
    assert si3_flux == 2
    assert fixed_ratio == 3
    assert builder.index("stellar_velocity") == velocity

    configuration = builder.build()
    assert configuration.parameter_names == (
        "stellar_velocity",
        "lya_log_total_flux",
        "si3_log_total_flux",
        "fixed_doublet_ratio",
    )
    np.testing.assert_allclose(
        configuration.base_parameters,
        [-20.0, -11.8, -13.0, 2.0],
    )
    assert tuple(type(parameter) for parameter in configuration.free_parameters) == (
        UniformParameter,
        UniformParameter,
        UniformParameter,
    )


def test_builder_supports_truncated_normal_and_configuration_snapshots():
    builder = InferenceConfigurationBuilder()
    builder.add_truncated_normal(
        "velocity",
        initial=-20.0,
        lower=-60.0,
        upper=20.0,
        mean=-18.0,
        standard_deviation=5.0,
    )
    first = builder.build()
    builder.add_uniform("width", 40.0, 5.0, 100.0)
    second = builder.build()

    assert isinstance(first.free_parameters[0], TruncatedNormalParameter)
    assert first.parameter_names == ("velocity",)
    assert second.parameter_names == ("velocity", "width")


def test_builder_rejects_ambiguous_or_invalid_parameters_without_mutation():
    builder = InferenceConfigurationBuilder()
    builder.add_uniform("velocity", -20.0, -60.0, 20.0)

    with pytest.raises(ValueError, match="already defined"):
        builder.add_fixed("velocity", -20.0)
    with pytest.raises(ValueError, match="within its bounds"):
        builder.add_uniform("width", 200.0, 5.0, 100.0)
    with pytest.raises(ValueError, match="must be finite"):
        builder.add_fixed("bad", np.nan)
    with pytest.raises(KeyError, match="unknown parameter"):
        builder.index("missing")

    assert builder.parameter_names == ("velocity",)
    assert builder.build().base_parameters == (-20.0,)


def test_builder_requires_at_least_one_free_parameter():
    builder = InferenceConfigurationBuilder()
    builder.add_fixed("constant", 1.0)

    with pytest.raises(ValueError, match="at least one free parameter"):
        builder.build()
