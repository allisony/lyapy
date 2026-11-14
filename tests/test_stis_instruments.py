from importlib import resources

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from lyapy.inference import (
    build_e140m_observation_operator,
    build_g140l_observation_operator,
    build_g140m_observation_operator,
    build_stis_observation_operator,
)
from lyapy.observations import (
    STIS_INSTRUMENT_CONFIGURATIONS,
    StisTargetSpecification,
    TabulatedLsf,
    get_stis_instrument_configuration,
    load_stis_lsf,
)


def test_instrument_registry_resolves_central_wavelength_suffixes():
    assert set(STIS_INSTRUMENT_CONFIGURATIONS) == {
        "E140M",
        "G140M",
        "G140L",
        "E140H",
        "E230M",
        "E230H",
        "G230L",
        "G230M",
    }
    e140m = get_stis_instrument_configuration("E140M-1425")
    g140m = get_stis_instrument_configuration("G140M/1222")

    assert e140m.profile_column("0.2×0.2") == 3
    assert g140m.profile_column('52X0.2"') == 2
    assert e140m.reference_wavelength == 1200.0
    assert e140m.select_calibration(1490.0).reference_wavelength == 1500.0
    assert g140m.as_dict()["calibrations"][0]["resource"].endswith(
        "LSF_G140M_1200.txt"
    )
    with pytest.raises(ValueError, match="unsupported"):
        get_stis_instrument_configuration("G430M")


def test_bundled_g140m_lsf_selects_the_requested_aperture_column():
    lsf = load_stis_lsf("G140M-1222", "52x0.2")
    package = resources.files("lyapy")
    with resources.as_file(
        package.joinpath("data/stis_lsf/LSF_G140M_1200.txt")
    ) as path:
        table = np.loadtxt(path, skiprows=2)

    np.testing.assert_array_equal(lsf.offset, table[:, 0])
    np.testing.assert_array_equal(lsf.profile, table[:, 2])
    with pytest.raises(ValueError, match="available apertures"):
        load_stis_lsf("G140M", "0.2x0.2")


def test_all_bundled_uv_lsf_tables_are_numerical_and_selectable():
    for configuration in STIS_INSTRUMENT_CONFIGURATIONS.values():
        aperture = configuration.apertures[0]
        profile_column = configuration.profile_column(aperture)
        for calibration in configuration.calibrations:
            lsf = load_stis_lsf(
                configuration.grating,
                aperture,
                wavelength=calibration.reference_wavelength,
            )
            package = resources.files("lyapy")
            with resources.as_file(
                package.joinpath(calibration.resource)
            ) as path:
                table = np.loadtxt(path, skiprows=2)
            np.testing.assert_array_equal(lsf.offset, table[:, 0])
            np.testing.assert_array_equal(
                lsf.profile,
                table[:, profile_column],
            )


def test_g140l_has_a_bundled_default_and_accepts_an_override(tmp_path):
    bundled = load_stis_lsf("G140L", "52x0.2", wavelength=1215.67)
    assert bundled.profile.size > 500
    target = StisTargetSpecification(
        name="synthetic G140L target",
        spectrum_resource="not-loaded.fits",
        grating="G140L-1425",
        aperture="52x0.2",
        wavelength_frame="heliocentric",
        stellar_radial_velocity=0.0,
        continuum_windows=((1190.0, 1200.0), (1250.0, 1260.0)),
        fit_window=(1210.0, 1220.0),
    )
    assert target.as_dict()["lsf_profile_column"] == 2
    assert target.as_dict()["lsf_resource"].endswith("LSF_G140L_1200.txt")

    path = tmp_path / "g140l_lsf.dat"
    path.write_text(
        "custom G140L test table\n"
        "relative_pixel 52x0.2\n"
        "-1.0 0.1\n"
        " 0.0 1.0\n"
        " 1.0 0.1\n",
        encoding="utf-8",
    )
    lsf = load_stis_lsf(
        "G140L",
        "52x0.2",
        path=path,
        profile_column=1,
    )

    np.testing.assert_allclose(lsf.offset, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(lsf.profile, [0.1, 1.0, 0.1])


def test_grating_builders_share_flux_conserving_operator_conventions():
    wavelength = np.linspace(1214.8, 1216.3, 41)
    e140m = build_e140m_observation_operator(
        wavelength,
        oversample=4,
        model_padding=0.5,
    )
    e140m_generic = build_stis_observation_operator(
        wavelength,
        grating="E140M",
        aperture="0.2x0.2",
        oversample=4,
        model_padding=0.5,
    )
    g140m = build_g140m_observation_operator(
        wavelength,
        aperture="52x0.2",
        oversample=4,
        model_padding=0.5,
    )

    np.testing.assert_array_equal(
        e140m.convolution_kernel,
        e140m_generic.convolution_kernel,
    )
    assert g140m.convolution_kernel.size > e140m.convolution_kernel.size
    np.testing.assert_allclose(np.sum(g140m.convolution_kernel), 1.0)
    _, detector_flux = g140m.apply(np.ones(g140m.wavelength.size))
    np.testing.assert_allclose(detector_flux, 1.0, atol=2e-15)


def test_all_grating_builders_accept_an_explicit_tabulation_override():
    wavelength = np.linspace(1210.0, 1220.0, 31)
    offsets = np.linspace(-3.0, 3.0, 61)
    lsf = TabulatedLsf(offsets, np.exp(-0.5 * offsets**2))
    builders = (
        build_e140m_observation_operator,
        build_g140m_observation_operator,
        build_g140l_observation_operator,
    )

    dispersion = np.median(np.diff(wavelength))
    for builder in builders:
        operator = builder(
            wavelength,
            lsf=lsf,
            oversample=4,
            model_padding=1.0,
        )
        np.testing.assert_allclose(operator.lsf_offsets, offsets * dispersion)
    with pytest.raises(TypeError, match="TabulatedLsf"):
        build_g140l_observation_operator(
            wavelength,
            lsf=(offsets, lsf.profile),
        )
