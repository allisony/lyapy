import numpy as np

from lyapy import lyapy
from lyapy.instrument import gaussian_kernel, interpolate_stis_lsf


def test_gaussian_convolution_kernel_is_approximately_normalized_and_symmetric():
    wavelength = np.linspace(1214.5, 1216.5, 1001)
    kernel = gaussian_kernel(wavelength, fwhm=1215.67 / 45_800.0)

    assert kernel.size % 2 == 1
    # make_kernel appends one tail sample after normalization, changing the sum
    # by less than one part per million for this representative STIS grid.
    np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(kernel, kernel[::-1], rtol=0.0, atol=1e-15)


def test_reference_stis_lsf_is_normalized(legacy_reference):
    kernel = legacy_reference["lsf_kernel"]

    assert kernel.size % 2 == 1
    assert np.all(kernel >= 0.0)
    np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=0.0, atol=1e-14)


def test_legacy_stis_wrapper_matches_public_api():
    original_wave = np.linspace(-4.0, 4.0, 17)
    original_lsf = np.exp(-0.5 * original_wave**2)
    data_wave = np.linspace(1214.5, 1216.5, 101)

    np.testing.assert_array_equal(
        lyapy.ready_stis_lsf(original_wave, original_lsf, 0.05, data_wave),
        interpolate_stis_lsf(original_wave, original_lsf, 0.05, data_wave),
    )
