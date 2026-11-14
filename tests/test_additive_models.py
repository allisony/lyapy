import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lyapy.inference.additive import (
    augment_configuration_with_background_and_airglow,
)
from lyapy.inference.common import InferenceConfiguration, UniformParameter
from lyapy.inference.composable import prepare_observed_spectrum_data
from lyapy.inference.dynesty_backend import build_dynesty_functions
from lyapy.jax_additive import (
    AffineBackgroundComponent,
    AffineContinuumComponent,
    CompositeAdditiveModel,
    GaussianAirglowComponent,
)
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    GaussianEmissionComponent,
    HydrogenDeuteriumAbsorber,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator
from lyapy.observations import ObservedSpectrum


def make_model():
    centers = np.linspace(1213.0, 1218.0, 101)
    offsets = np.linspace(-0.5, 0.5, 401)
    lsf = np.exp(-0.5 * (offsets / 0.12) ** 2)
    operator = build_high_resolution_operator(
        centers,
        offsets,
        lsf,
        oversample=8,
        model_padding=1.5,
    )
    configuration = InferenceConfiguration(
        base_parameters=(0.0, -12.0, 100.0),
        free_parameters=(
            UniformParameter("stellar_velocity", 0, -50.0, 50.0),
            UniformParameter("log_total_flux", 1, -14.0, -10.0),
            UniformParameter("stellar_fwhm", 2, 20.0, 300.0),
        ),
        parameter_names=(
            "stellar_velocity",
            "log_total_flux",
            "stellar_fwhm",
        ),
    )
    emission = CompositeEmissionModel(
        (GaussianEmissionComponent(1215.67, 0, 2),),
        1,
    )
    return centers, configuration, ComposableBinnedModel(emission, operator)


def test_additive_components_are_binned_without_point_source_convolution():
    centers, _, source_model = make_model()
    additive = CompositeAdditiveModel(
        (
            AffineBackgroundComponent(
                1215.5,
                3,
                4,
                flux_scale=1e-15,
                wavelength_scale=2.0,
            ),
            GaussianAirglowComponent(
                1215.58,
                5,
                7,
                6,
                flux_scale=2e-15,
            ),
        )
    )
    model = ComposableBinnedModel(
        source_model.emission_model,
        source_model.observation_operator,
        additive_model=additive,
    )
    parameters = jnp.array(
        [0.0, -12.0, 100.0, 0.7, -0.2, 1.4, 4.0, 260.0]
    )

    result = model(centers, parameters, np.array([1.0]))
    expected_additive = source_model.observation_operator.bin(
        additive.evaluate(
            source_model.observation_operator.wavelength,
            parameters,
        ).total
    )

    np.testing.assert_allclose(result.convolved.additive, expected_additive)
    np.testing.assert_allclose(
        result.convolved.observed,
        result.convolved.point_source + result.convolved.additive,
    )
    assert result.additive.component_fluxes.shape[0] == 2


def test_affine_background_is_detector_bin_averaged_at_pixel_centers():
    centers, _, source_model = make_model()
    component = AffineBackgroundComponent(
        1215.5,
        0,
        1,
        flux_scale=1e-15,
        wavelength_scale=2.0,
    )
    parameters = jnp.array([1.5, -0.4])
    high_resolution = component.flux_density(
        source_model.observation_operator.wavelength,
        parameters,
    )
    detector = source_model.observation_operator.bin(high_resolution)
    expected = 1e-15 * (1.5 - 0.4 * (centers - 1215.5) / 2.0)

    np.testing.assert_allclose(detector, expected, rtol=3e-7, atol=1e-28)


def test_airglow_uses_signed_integrated_flux_and_fitted_center_offset():
    _, _, source_model = make_model()
    component = GaussianAirglowComponent(
        1215.58,
        0,
        2,
        1,
        flux_scale=2e-15,
    )
    wavelength = jnp.asarray(source_model.observation_operator.wavelength)
    positive = component.flux_density(
        wavelength,
        jnp.array([1.5, 8.0, 260.0]),
    )
    negative = component.flux_density(
        wavelength,
        jnp.array([-1.5, 8.0, 260.0]),
    )
    expected_center = 1215.58 * (1.0 + 8.0 / 299792.458)

    np.testing.assert_allclose(negative, -positive)
    np.testing.assert_allclose(
        np.trapezoid(positive, wavelength),
        3e-15,
        rtol=2e-10,
    )
    assert wavelength[np.argmax(positive)] == pytest.approx(
        expected_center,
        abs=np.median(np.diff(wavelength)),
    )


def test_stellar_continuum_is_absorbed_but_detector_background_is_not():
    centers, _, source_model = make_model()
    absorption = CompositeAbsorptionModel(
        (HydrogenDeuteriumAbsorber(3, 4, 5),)
    )
    parameters = jnp.array([0.0, -30.0, 100.0, 19.0, 12.0, 0.0, 1.0])
    background = CompositeAdditiveModel(
        (AffineBackgroundComponent(1215.5, 6, flux_scale=1e-15),)
    )
    continuum = CompositeAdditiveModel(
        (AffineContinuumComponent(1215.5, 6, flux_scale=1e-15),)
    )
    background_model = ComposableBinnedModel(
        source_model.emission_model,
        source_model.observation_operator,
        absorption_model=absorption,
        additive_model=background,
    )
    continuum_model = ComposableBinnedModel(
        source_model.emission_model,
        source_model.observation_operator,
        absorption_model=absorption,
        continuum_model=continuum,
    )

    detector_background = background_model(
        centers, parameters, np.ones(1)
    ).convolved.observed
    stellar_continuum = continuum_model(
        centers, parameters, np.ones(1)
    ).convolved.observed
    center_index = np.argmin(np.abs(centers - 1215.67))

    assert detector_background[center_index] > 0.9e-15
    assert stellar_continuum[center_index] < 0.1e-15


def test_configuration_augmentation_is_explicit_and_differentiable():
    centers, configuration, model = make_model()
    augmented, fitted_model = (
        augment_configuration_with_background_and_airglow(
            configuration,
            model,
            airglow_center=1215.58,
            baseline_flux_density_scale=1e-15,
            airglow_flux_scale=2e-15,
            base_airglow_fwhm=260.0,
            airglow_fwhm_bounds=(100.0, 500.0),
            prefix="exposure_1",
            baseline_wavelength_scale=2.0,
        )
    )
    expected_names = (
        "exposure_1_background_intercept",
        "exposure_1_background_slope",
        "exposure_1_airglow_scaled_total_flux",
        "exposure_1_airglow_velocity_offset",
        "exposure_1_airglow_fwhm",
    )

    assert augmented.parameter_names[-5:] == expected_names
    assert augmented.free_parameters[-2].mean == 0.0
    assert augmented.free_parameters[-2].standard_deviation == 10.0
    assert fitted_model.additive_model is not None

    parameters = jnp.asarray(augmented.base_parameters)
    jacobian = jax.jacfwd(
        lambda values: fitted_model(
            centers,
            values,
            np.array([1.0]),
        ).convolved.observed
    )(parameters)
    nuisance_sensitivity = np.max(np.abs(np.asarray(jacobian)[:, -5:]), axis=0)

    assert np.all(np.isfinite(jacobian))
    assert np.all(nuisance_sensitivity > 0.0)


def test_augmentation_can_place_an_astrophysical_continuum_before_ism():
    _, configuration, model = make_model()
    augmented, fitted_model = (
        augment_configuration_with_background_and_airglow(
            configuration,
            model,
            airglow_center=1215.58,
            baseline_flux_density_scale=1e-15,
            airglow_flux_scale=2e-15,
            base_airglow_fwhm=260.0,
            airglow_fwhm_bounds=(100.0, 500.0),
            baseline_kind="stellar_continuum",
        )
    )

    assert augmented.parameter_names[-5:-3] == (
        "continuum_intercept",
        "continuum_slope",
    )
    assert fitted_model.continuum_model is not None
    assert len(fitted_model.additive_model.components) == 1


def test_joint_exposures_can_share_one_airglow_width_parameter():
    _, configuration, first_base_model = make_model()
    _, _, second_base_model = make_model()
    arguments = dict(
        airglow_center=1215.58,
        baseline_flux_density_scale=1e-15,
        airglow_flux_scale=2e-15,
        base_airglow_fwhm=260.0,
        airglow_fwhm_bounds=(100.0, 500.0),
        shared_airglow_fwhm_name="g140l_airglow_fwhm",
    )
    first_configuration, first_model = (
        augment_configuration_with_background_and_airglow(
            configuration,
            first_base_model,
            prefix="first",
            **arguments,
        )
    )
    second_configuration, second_model = (
        augment_configuration_with_background_and_airglow(
            first_configuration,
            second_base_model,
            prefix="second",
            **arguments,
        )
    )
    first_airglow = first_model.additive_model.components[-1]
    second_airglow = second_model.additive_model.components[-1]

    assert second_configuration.parameter_names.count(
        "g140l_airglow_fwhm"
    ) == 1
    assert first_airglow.fwhm_index == second_airglow.fwhm_index
    assert len(second_configuration.parameter_names) == (
        len(configuration.parameter_names) + 9
    )


def test_augmentation_rejects_hidden_or_duplicate_scientific_choices():
    _, configuration, model = make_model()
    arguments = dict(
        airglow_center=1215.58,
        baseline_flux_density_scale=1e-15,
        airglow_flux_scale=2e-15,
        base_airglow_fwhm=260.0,
        airglow_fwhm_bounds=(100.0, 500.0),
    )
    augmented, fitted_model = (
        augment_configuration_with_background_and_airglow(
            configuration,
            model,
            **arguments,
        )
    )

    with pytest.raises(ValueError, match="already contains an additive model"):
        augment_configuration_with_background_and_airglow(
            augmented,
            fitted_model,
            **arguments,
        )
    with pytest.raises(ValueError, match="baseline_flux_density_scale"):
        augment_configuration_with_background_and_airglow(
            configuration,
            model,
            **{**arguments, "baseline_flux_density_scale": 0.0},
        )


def test_raw_spectrum_preparation_preserves_native_grid_and_nuisance_flux():
    wavelength = np.arange(1204.0, 1225.0)
    error = np.full(wavelength.shape, 0.2)
    error[11] = 0.0
    quality = np.ones(wavelength.shape, dtype=bool)
    quality[13] = False
    spectrum = ObservedSpectrum(
        wavelength,
        0.3 * wavelength - 360.0,
        error,
        quality_mask=quality,
        metadata={"rootname": "synthetic"},
    )

    data = prepare_observed_spectrum_data(
        spectrum,
        (1205.0, 1223.0),
        exclusion_windows=((1206.0, 1207.0),),
    )

    np.testing.assert_array_equal(data.wavelength, np.arange(1205.0, 1224.0))
    np.testing.assert_array_equal(
        data.flux,
        spectrum.flux[(wavelength >= 1205.0) & (wavelength <= 1223.0)],
    )
    assert data.fitted_pixels == 15
    assert np.all(data.error > 0.0)
    assert data.metadata["continuum_subtracted"] is False
    assert data.metadata["airglow_subtracted"] is False
    assert data.metadata["nonpositive_error_pixels_excluded"] == 1
    assert data.metadata["excluded_windows_angstrom"] == [[1206.0, 1207.0]]


def test_additive_parameters_participate_in_shared_inference_likelihood():
    centers, configuration, model = make_model()
    augmented, fitted_model = (
        augment_configuration_with_background_and_airglow(
            configuration,
            model,
            airglow_center=1215.58,
            baseline_flux_density_scale=1e-15,
            airglow_flux_scale=2e-15,
            base_airglow_fwhm=260.0,
            airglow_fwhm_bounds=(100.0, 500.0),
            baseline_wavelength_scale=2.0,
            base_baseline_intercept=0.7,
            base_baseline_slope=-0.2,
            base_airglow_scaled_total_flux=1.4,
            base_airglow_velocity_offset=4.0,
        )
    )
    full_truth = np.asarray(augmented.base_parameters)
    observed = np.asarray(
        fitted_model(
            centers,
            jnp.asarray(full_truth),
            np.ones(1),
        ).convolved.observed
    )
    data = prepare_observed_spectrum_data(
        ObservedSpectrum(
            centers,
            observed,
            np.full(centers.shape, 2e-16),
        ),
        (centers[0], centers[-1]),
    )
    log_likelihood, _ = build_dynesty_functions(
        data,
        augmented,
        forward_model=fitted_model,
    )
    free_truth = np.asarray(
        [full_truth[item.index] for item in augmented.free_parameters]
    )
    wrong_airglow = free_truth.copy()
    airglow_position = next(
        index
        for index, item in enumerate(augmented.free_parameters)
        if item.name == "airglow_scaled_total_flux"
    )
    wrong_airglow[airglow_position] = -1.0

    assert log_likelihood(free_truth) > log_likelihood(wrong_airglow)
