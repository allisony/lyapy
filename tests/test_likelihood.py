import numpy as np

from lyapy import fitting


def constant_model(x, resolution, parameters, variables, lnlike=True):
    del resolution, variables, lnlike
    return [np.full_like(x[0], parameters[0], dtype=float)]


def test_gaussian_log_likelihood_matches_direct_calculation():
    x = [np.array([1.0, 2.0, 3.0])]
    y = [np.array([1.5, 2.5, 1.0])]
    yerr = [np.array([0.2, 0.3, 0.4])]
    mask = [[False, False, False]]
    theta = [2.0]

    actual = fitting.lnlike(
        theta,
        x,
        y,
        yerr,
        resolution=[None],
        variables={},
        model_function=constant_model,
        mask_data=mask,
    )
    residual = y[0] - theta[0]
    expected = -0.5 * np.sum(
        residual**2 / yerr[0] ** 2 + np.log(2.0 * np.pi * yerr[0] ** 2)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_prior_rejects_values_outside_bounds():
    actual = fitting.lnprior(
        theta=[2.1],
        minmax=[[0.0, 2.0]],
        prior_Gauss=[False],
        prior_list=[0],
    )

    assert actual == -np.inf


def test_gaussian_prior_matches_direct_calculation():
    actual = fitting.lnprior(
        theta=[1.5],
        minmax=[[0.0, 2.0]],
        prior_Gauss=[True],
        prior_list=[[1.0, 0.25]],
    )

    np.testing.assert_allclose(actual, -0.5 * ((1.5 - 1.0) / 0.25) ** 2)
