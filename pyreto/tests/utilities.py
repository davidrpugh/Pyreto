from scipy import stats


def test_scaling_exponent_estimation(desired_alpha, result, size=0.01):
    """
    Test whether the desired alpha lies within some specified confidence
    interval of the estimated scaling exponent.

    """
    critical_value = stats.norm.ppf(size / 2)  # this is negative!
    alpha_hat, alpha_se = result.params['alpha'], result.standard_errors['alpha']
    lower = alpha_hat + critical_value * alpha_se
    upper = alpha_hat - critical_value * alpha_se
    assert lower <= desired_alpha <= upper
