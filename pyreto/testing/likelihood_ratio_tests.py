from scipy import stats


def vuong_test(result1, result2):
    """Vuong likelihood ratio test."""
    assert result1.n_tail == result2.n_tail
    # compute the components of the vuong statistic
    pointwise_log_likelihood = result1.log_likelihood - result2.log_likelihood
    loglikelihood_ratio = pointwise_log_likelihood.sum()
    omega = pointwise_log_likelihood.std()
    n = pointwise_log_likelihood.size

    # compute the vuong statistic and p-values
    vuong_statistic = loglikelihood_ratio / (n**0.5 * omega)
    p1 = stats.norm.cdf(vuong_statistic)
    p2 = 2 * p1 if p1 < 0.5 else 2 * (1 - p1)

    return vuong_statistic, p1, p2
