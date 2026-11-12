import numpy as np
import pytest

from lyapy import lyapy
from lyapy.priors import legacy_bounded_log_prior


def test_legacy_prior_combines_gaussian_and_named_log_weight():
    actual = legacy_bounded_log_prior(
        values=[1.5, 12.0],
        bounds=[[0.0, 2.0], [8.0, 20.0]],
        gaussian_flags=[True, False],
        gaussian_parameters=[[1.0, 0.25], 0],
        parameter_names=["amplitude", "h1_b"],
        log_weight_parameters=("h1_b",),
    )
    expected = -0.5 * ((1.5 - 1.0) / 0.25) ** 2 + np.log(12.0)

    np.testing.assert_allclose(actual, expected)


def test_legacy_prior_rejects_bad_shapes_and_out_of_bounds_values():
    assert legacy_bounded_log_prior(
        [3.0],
        [[0.0, 2.0]],
        [False],
        [0],
    ) == -np.inf
    with pytest.raises(ValueError, match="bounds"):
        legacy_bounded_log_prior([1.0], [[0.0, 2.0], [0.0, 3.0]], [False], [0])


def test_all_legacy_prior_variants_apply_h1_b_weight_by_name():
    cases = (
        (lyapy.lnprior, 10, 7),
        (lyapy.lnprior_g140l, 16, 7),
        (lyapy.lnprior_voigt, 12, 5),
        (lyapy.lnprior_voigt_cii, 6, 4),
        (lyapy.lnprior_voigt_rev, 11, 5),
    )
    for prior_function, size, h1_b_index in cases:
        values = np.ones(size)
        values[h1_b_index] = 12.0
        bounds = np.column_stack((np.zeros(size), np.full(size, 20.0)))
        actual = prior_function(values, bounds, np.zeros(size, dtype=bool), [0] * size)

        np.testing.assert_allclose(actual, np.log(12.0))
