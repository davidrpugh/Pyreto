from scipy import stats

from .. import distributions


def test_scaling_exponent_estimation(desired_alpha, result, size=0.01):
    """
    Test whether the desired alpha lies within some specified confidence
    interval of the estimated scaling exponent.

    """
    critical_value = stats.norm.ppf(size / 2)  # this is negative!
    b_hat, b_se = result.params['b'], result.standard_errors['b']
    lower = b_hat + critical_value * b_se
    upper = b_hat - critical_value * b_se
    assert lower <= desired_alpha - 1 <= upper


def replicate_estimation_results(data, reported_xmin, method='brute',
                                 discrete=False, approx=False):
    """
    Attempt to replicate parameter estimation results reported in Clauset,
    Shalizi, and Newman (2009) for a given data set.

    Parameters
    ----------
    data : pandas.Series
    reported_xmin : float
    discrete : Boolean (default=False)
    approx : Boolean (default=False)

    Returns
    -------
    result1 :
    result2 :

    """
    # estimate scaling exponent taking reported xmin as given...
    result1 = distributions.Pareto.fit(data, scale=reported_xmin,
                                       discrete=discrete, approx=approx)

    # jointly estimate scaling exponent and threshold...
    result2 = distributions.Pareto.fit(data, scale=None, method=method)

    return result1, result2


def replicate_goodness_of_fit_results(result, data, method='brute',
                                      discrete=False, approx=False,
                                      replications=1000):
    """
    Attempt to replicate goodness of fit test results reported in Clauset,
    Shalizi, and Newman (2009) for a given data set.

    Parameters
    ----------
    data : pandas.Series
    reported_xmin : float
    discrete : boolean (default=False)
    approx : boolean (default=False)
    replications : int

    Returns
    -------
    pvalue : float
    Ds : numpy.ndarray

    """
    kwargs = {'method': method, 'discrete': discrete, 'approx': approx,
              'replications': replications}
    pvalue, Ds = distributions.Pareto.test_goodness_of_fit(result, data, **kwargs)
    return pvalue, Ds


def replicate_voung_test_results(result, data):
    pass
