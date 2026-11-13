import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference.common import InferenceConfigurationBuilder
from lyapy.inference.multiplet import make_multiplet_ratio_hypotheses
from lyapy.jax_components import (
    AtomicTransitionAbsorber,
    CompositeAbsorptionModel,
)
from lyapy.jax_instrument import build_high_resolution_operator


def make_problem(*, ratio_is_free=True, absorbed=False, voigt=False):
    builder = InferenceConfigurationBuilder()
    velocity = builder.add_uniform("stellar_velocity", -10.0, -50.0, 30.0)
    flux = builder.add_uniform("log_total_flux", -12.0, -13.0, -11.0)
    gaussian_width = builder.add_uniform("gaussian_fwhm", 45.0, 5.0, 100.0)
    if ratio_is_free:
        ratio = builder.add_uniform(
            "log_nv_1242_to_1238",
            np.log10(0.5),
            -1.0,
            0.3,
        )
    else:
        ratio = builder.add_fixed("log_nv_1242_to_1238", np.log10(0.5))
    lorentz_width = None
    if voigt:
        lorentz_width = builder.add_uniform(
            "lorentz_fwhm",
            8.0,
            1.0,
            40.0,
        )
    absorption_model = None
    if absorbed:
        ism_column = builder.add_fixed("nv_ism_log_column", 13.5)
        ism_doppler_b = builder.add_fixed("nv_ism_doppler_b", 10.0)
        ism_velocity = builder.add_fixed("nv_ism_velocity", 2.0)
        absorption_model = CompositeAbsorptionModel(
            (
                AtomicTransitionAbsorber(
                    "n_v_1238",
                    ism_column,
                    ism_doppler_b,
                    ism_velocity,
                ),
                AtomicTransitionAbsorber(
                    "n_v_1242",
                    ism_column,
                    ism_doppler_b,
                    ism_velocity,
                ),
            )
        )
    configuration = builder.build()
    wavelength = np.linspace(1237.5, 1244.0, 651)
    offsets = np.linspace(-0.15, 0.15, 301)
    lsf = np.exp(-0.5 * (offsets / 0.025) ** 2)
    operator = build_high_resolution_operator(
        wavelength,
        offsets,
        lsf,
        oversample=4,
        model_padding=0.3,
    )
    hypotheses = make_multiplet_ratio_hypotheses(
        "n_v_1238_1242",
        operator,
        configuration,
        velocity_index=velocity,
        log_total_flux_index=flux,
        gaussian_fwhm_index=gaussian_width,
        lorentz_fwhm_index=lorentz_width,
        log_relative_weight_indices=(ratio,),
        fixed_relative_weights=(2.0, 1.0),
        absorption_model=absorption_model,
    )
    return wavelength, configuration, hypotheses


def prediction(hypothesis, wavelength, parameters):
    return hypothesis.forward_model(
        wavelength,
        jnp.asarray(parameters),
        np.array([1.0]),
    ).convolved.observed


def test_fixed_and_free_ratio_models_match_at_the_fixed_ratio():
    wavelength, configuration, hypotheses = make_problem()

    fixed = prediction(
        hypotheses.fixed_ratio,
        wavelength,
        hypotheses.fixed_ratio.configuration.base_parameters,
    )
    free = prediction(
        hypotheses.free_ratio,
        wavelength,
        configuration.base_parameters,
    )

    np.testing.assert_allclose(fixed, free, rtol=2e-15, atol=1e-30)
    assert tuple(hypotheses) == hypotheses.hypotheses
    assert [
        parameter.name
        for parameter in hypotheses.fixed_ratio.configuration.free_parameters
    ] == ["stellar_velocity", "log_total_flux", "gaussian_fwhm"]


def test_free_ratio_changes_multiplet_shape_and_is_serialized():
    wavelength, configuration, hypotheses = make_problem()
    equal_ratio_parameters = list(configuration.base_parameters)
    ratio_index = configuration.parameter_names.index(
        "log_nv_1242_to_1238"
    )
    equal_ratio_parameters[ratio_index] = 0.0

    fixed = prediction(
        hypotheses.fixed_ratio,
        wavelength,
        hypotheses.fixed_ratio.configuration.base_parameters,
    )
    free = prediction(
        hypotheses.free_ratio,
        wavelength,
        equal_ratio_parameters,
    )
    report = hypotheses.free_ratio.as_dict()

    assert not np.allclose(fixed, free, rtol=1e-4, atol=0.0)
    assert report["forward_model_specification"]["component_ratio"] == {
        "mode": "free",
        "log10_parameter_names": ["log_nv_1242_to_1238"],
    }
    assert hypotheses.fixed_ratio.as_dict()["forward_model_specification"][
        "component_ratio"
    ] == {"mode": "fixed", "relative_weights": [2.0, 1.0]}


def test_voigt_and_absorbed_multiplet_hypotheses_are_described():
    _, _, hypotheses = make_problem(absorbed=True, voigt=True)
    specification = hypotheses.free_ratio.as_dict()[
        "forward_model_specification"
    ]

    assert specification["profile"] == "voigt"
    assert specification["includes_absorption"]
    assert specification["line_group"]["key"] == "n_v_1238_1242"


def test_ratio_comparison_requires_explicit_valid_free_ratio_parameters():
    with pytest.raises(ValueError, match="must identify free parameters"):
        make_problem(ratio_is_free=False)

    builder = InferenceConfigurationBuilder()
    velocity = builder.add_uniform("velocity", 0.0, -10.0, 10.0)
    flux = builder.add_uniform("flux", -12.0, -13.0, -11.0)
    width = builder.add_uniform("width", 40.0, 5.0, 100.0)
    ratio = builder.add_uniform("ratio", -0.3, -1.0, 0.3)
    configuration = builder.build()
    wavelength = np.linspace(1206.2, 1206.8, 61)
    operator = build_high_resolution_operator(
        wavelength,
        np.linspace(-0.1, 0.1, 101),
        np.ones(101),
        oversample=2,
    )

    with pytest.raises(ValueError, match="multiple transitions"):
        make_multiplet_ratio_hypotheses(
            "si_iii_1206",
            operator,
            configuration,
            velocity_index=velocity,
            log_total_flux_index=flux,
            gaussian_fwhm_index=width,
            log_relative_weight_indices=(ratio,),
            fixed_relative_weights=(1.0,),
        )

    with pytest.raises(ValueError, match="distinct from profile parameters"):
        make_multiplet_ratio_hypotheses(
            "n_v_1238_1242",
            operator,
            configuration,
            velocity_index=velocity,
            log_total_flux_index=flux,
            gaussian_fwhm_index=ratio,
            log_relative_weight_indices=(ratio,),
            fixed_relative_weights=(2.0, 1.0),
        )
