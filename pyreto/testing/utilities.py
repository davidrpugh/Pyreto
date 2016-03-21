from scipy import stats

from .. import distributions
from . import likelihood_ratio_tests


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


def replicate_estimation_results(data, reported_xmin, **fit_kwargs):
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
    result1 = distributions.Pareto.fit(data, scale=reported_xmin, **fit_kwargs)

    # jointly estimate scaling exponent and threshold...
    result2 = distributions.Pareto.fit(data, scale=None, **fit_kwargs)

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


def replicate_voung_test_results(data, xmin):
    """
    Attempt to replicate Vuong likelihood ratio tests results reported in
    Clauset, Shalizi, and Newman (2009) for a given data set.

    Parameters
    ----------
    data : pandas.Series
    xmin : float

    Returns
    -------

    """
    pareto_fit = distributions.Pareto.fit(data, scale=xmin)
    expon_fit = distributions.Exponential.fit(data, floc=xmin)
    exponweib_fit = distributions.StretchedExponential.fit(data, floc=xmin)
    print(exponweib_fit.params)

    lognorm_fit = distributions.LogNormal.fit(data, floc=xmin)

    expon_result = likelihood_ratio_tests.vuong_test(pareto_fit, expon_fit)
    exponweib_result = likelihood_ratio_tests.vuong_test(pareto_fit, exponweib_fit)
    lognorm_result = likelihood_ratio_tests.vuong_test(pareto_fit, lognorm_fit)

    return {'expon': expon_result, 'exponweib': exponweib_result, 'lognorm': lognorm_result}
