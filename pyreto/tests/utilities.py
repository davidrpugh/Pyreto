from scipy import stats


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
