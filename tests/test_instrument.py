import numpy as np

from lyapy import lyapy


def test_gaussian_convolution_kernel_is_approximately_normalized_and_symmetric():
    wavelength = np.linspace(1214.5, 1216.5, 1001)
    kernel = lyapy.make_kernel(wavelength, fwhm=1215.67 / 45_800.0)

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
