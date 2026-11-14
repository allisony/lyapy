import numpy as np
import pytest

from lyapy.observations import (
    AYRES_PHOTERR_SCALE,
    GJ411_E140M_TARGET,
    GJ729_E140M_TARGET,
    ObservedSpectrum,
    TabulatedLsf,
    fit_continuum,
    load_gj411_observation,
    load_gj729_observation,
    prepare_stis_observation,
)


def test_packaged_gj411_observation_matches_frozen_reference(legacy_reference):
    observation = load_gj411_observation()

    assert observation.name == "GJ 411"
    np.testing.assert_array_equal(
        observation.continuum_coefficients,
        legacy_reference["continuum_coefficients"],
    )
    np.testing.assert_array_equal(observation.wavelength, legacy_reference["wave"])
    np.testing.assert_array_equal(observation.flux, legacy_reference["flux"])
    np.testing.assert_allclose(
        observation.error,
        legacy_reference["error"] * AYRES_PHOTERR_SCALE,
        rtol=1e-15,
    )
    np.testing.assert_array_equal(
        observation.convolution_kernel,
        legacy_reference["lsf_kernel"],
    )
    assert observation.uncertainty_scale == AYRES_PHOTERR_SCALE
    assert np.count_nonzero(~observation.likelihood_mask) == 7
    assert observation.metadata == GJ411_E140M_TARGET.as_dict()


def test_packaged_gj729_observation_uses_reviewed_target_recipe():
    observation = load_gj729_observation()

    assert observation.name == "GJ 729"
    assert observation.wavelength.size == 116
    assert observation.likelihood_mask.sum() == 108
    assert observation.uncertainty_scale == AYRES_PHOTERR_SCALE
    np.testing.assert_allclose(
        observation.raw_error * AYRES_PHOTERR_SCALE,
        observation.error,
    )
    assert observation.metadata == GJ729_E140M_TARGET.as_dict()
    assert observation.metadata["aperture"] == "0.2x0.2"
    assert observation.metadata["wavelength_frame"] == "heliocentric"
    assert observation.metadata["program"] == "HST GO 15071"


def test_generic_stis_preprocessing_subtracts_known_continuum():
    wavelength = np.linspace(0.0, 10.0, 101)
    continuum = 2.0 * wavelength + 3.0
    spectrum = ObservedSpectrum(
        wavelength,
        continuum,
        np.full(wavelength.shape, 0.1),
        name="synthetic",
    )
    lsf = TabulatedLsf(
        np.linspace(-2.0, 2.0, 41),
        np.exp(-0.5 * np.linspace(-2.0, 2.0, 41) ** 2),
    )

    observation = prepare_stis_observation(
        spectrum,
        lsf,
        continuum_windows=((0.0, 2.0), (8.0, 10.0)),
        fit_window=(4.0, 6.0),
        uncertainty_scale=2.0,
        excluded_windows=((4.8, 5.2),),
    )

    np.testing.assert_allclose(observation.flux, 0.0, atol=1e-14)
    np.testing.assert_allclose(observation.continuum_coefficients, [2.0, 3.0])
    np.testing.assert_allclose(np.sum(observation.convolution_kernel), 1.0)
    np.testing.assert_allclose(observation.error, 0.2)
    assert np.all(
        ~observation.likelihood_mask[
            (observation.wavelength >= 4.8)
            & (observation.wavelength <= 5.2)
        ]
    )


def test_continuum_fit_rejects_empty_windows():
    wavelength = np.linspace(0.0, 1.0, 11)
    spectrum = ObservedSpectrum(
        wavelength,
        np.ones(wavelength.shape),
        np.ones(wavelength.shape),
    )

    with pytest.raises(ValueError, match="enough samples"):
        fit_continuum(spectrum, ((2.0, 3.0),))


def test_observed_spectrum_requires_increasing_wavelengths():
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservedSpectrum([1.0, 0.0], [1.0, 1.0], [0.1, 0.1])
