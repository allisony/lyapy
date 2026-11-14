import numpy as np
import pytest
from astropy.io import fits

from lyapy.observations import (
    AYRES_PHOTERR_SCALE,
    GJ411_E140M_TARGET,
    GJ729_E140M_TARGET,
    ObservedSpectrum,
    TabulatedLsf,
    decode_stis_dq,
    fit_continuum,
    load_gj411_observation,
    load_gj729_observation,
    load_stis_x1d,
    prepare_stis_observation,
    stis_airglow_wavelength,
)


def _write_stis_x1d(path, wavelengths):
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    rows, pixels = wavelengths.shape
    flux = 2.0 * wavelengths + 3.0
    error = np.full(wavelengths.shape, 0.2)
    dq = np.zeros(wavelengths.shape, dtype=np.int32)
    error[0, 1] = 0.0
    dq[0, 2] = 4
    primary = fits.PrimaryHDU()
    primary.header.update(
        {
            "INSTRUME": "STIS",
            "DETECTOR": "FUV-MAMA",
            "OPT_ELEM": "G140M",
            "APERTURE": "52X0.2",
            "CENWAVE": 1222,
            "ROOTNAME": "synthetic",
            "TARGNAME": "TEST-STAR",
            "PROPOSID": 12345,
            "HELCORR": "COMPLETE",
            "CAL_VER": "test",
        }
    )
    columns = fits.ColDefs(
        [
            fits.Column(
                name="SPORDER",
                format="J",
                array=np.arange(1, rows + 1),
            ),
            fits.Column(
                name="NELEM",
                format="J",
                array=np.full(rows, pixels),
            ),
            fits.Column(
                name="WAVELENGTH",
                format=f"{pixels}D",
                unit="Angstroms",
                array=wavelengths,
            ),
            fits.Column(
                name="FLUX",
                format=f"{pixels}D",
                unit="erg/s/cm**2/Angstrom",
                array=flux,
            ),
            fits.Column(
                name="ERROR",
                format=f"{pixels}D",
                unit="erg/s/cm**2/Angstrom",
                array=error,
            ),
            fits.Column(
                name="DQ",
                format=f"{pixels}J",
                array=dq,
            ),
        ]
    )
    science = fits.BinTableHDU.from_columns(columns, name="SCI")
    science.header["SDQFLAGS"] = 4
    science.header["V_HELIO"] = 12.5
    science.header["EXPTIME"] = 1000.0
    fits.HDUList([primary, science]).writeto(path)


def test_stis_x1d_loader_preserves_headers_units_and_quality_mask(tmp_path):
    path = tmp_path / "synthetic_x1d.fits"
    _write_stis_x1d(path, [[1200.0, 1201.0, 1202.0, 1203.0]])

    spectrum = load_stis_x1d(
        path,
        name="Synthetic target",
        metadata_overrides={"coadded": True},
    )

    assert spectrum.name == "Synthetic target"
    np.testing.assert_array_equal(
        spectrum.quality_mask,
        [True, False, False, True],
    )
    assert spectrum.metadata["grating"] == "G140M"
    assert spectrum.metadata["aperture"] == "52X0.2"
    assert spectrum.metadata["wavelength_frame"] == "heliocentric"
    assert spectrum.metadata["heliocentric_velocity_km_s"] == 12.5
    assert spectrum.metadata["coadded"] is True
    assert spectrum.metadata["excluded_quality_pixels"] == 2
    assert spectrum.metadata["dq_flag_counts"] == {"4": 1}
    assert spectrum.metadata["extraction_offset_pixels"] is None
    assert spectrum.metadata["wavelength_unit"] == "Angstroms"
    np.testing.assert_array_equal(spectrum.data_quality, [0, 0, 4, 0])


def test_stis_dq_decoder_and_airglow_location_are_explicit():
    assert decode_stis_dq(32 | 16) == (
        (16, "Dark rate more than 5 sigma above the median"),
        (32, "Large blemish below 40% flat-field response (repeller wire)"),
    )
    assert decode_stis_dq(0) == ()
    with pytest.raises(ValueError, match="non-negative"):
        decode_stis_dq(-1)

    # A positive heliocentric correction shifts stationary terrestrial
    # airglow blueward on the corrected wavelength grid.
    assert stis_airglow_wavelength(10.0) < 1215.67


def test_stis_x1d_loader_selects_echelle_order_by_wavelength(tmp_path):
    path = tmp_path / "synthetic_x1d.fits"
    _write_stis_x1d(
        path,
        [
            [1200.0, 1201.0, 1202.0, 1203.0],
            [1300.0, 1301.0, 1302.0, 1303.0],
        ],
    )

    with pytest.raises(ValueError, match="multi-order"):
        load_stis_x1d(path)
    spectrum = load_stis_x1d(path, wavelength=1301.0)

    np.testing.assert_array_equal(
        spectrum.wavelength,
        [1300.0, 1301.0, 1302.0, 1303.0],
    )
    assert spectrum.metadata["spectral_order"] == 2


def test_packaged_gj411_observation_matches_frozen_reference(legacy_reference):
    observation = load_gj411_observation()

    assert observation.name == "GJ 411"
    np.testing.assert_allclose(
        observation.continuum_coefficients,
        legacy_reference["continuum_coefficients"],
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(observation.wavelength, legacy_reference["wave"])
    np.testing.assert_allclose(
        observation.flux,
        legacy_reference["flux"],
        rtol=1e-12,
        atol=1e-25,
    )
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
