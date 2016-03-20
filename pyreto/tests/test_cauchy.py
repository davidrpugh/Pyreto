import numpy as np
import pandas as pd
from scipy import stats

from .. import distributions
from . import utilities

# Fit the pareto distribution to Cauchy data
prng = np.random.RandomState(1234)
CAUCHY_RVS = stats.cauchy.rvs(size=2500, random_state=prng)
CAUCHY_POSITIVE_TAIL = pd.Series(CAUCHY_RVS[CAUCHY_RVS > 0], name='samples')
CAUCHY_NEGATIVE_TAIL = pd.Series(-CAUCHY_RVS[CAUCHY_RVS < 0], name='samples')

QUANTILE, METHOD = 0.99, 'bounded'
RESULT1 = distributions.Pareto.fit(CAUCHY_POSITIVE_TAIL, scale=None,
                                   quantile=QUANTILE, method=METHOD)
RESULT2 = distributions.Pareto.fit(CAUCHY_NEGATIVE_TAIL, scale=None,
                                   quantile=QUANTILE, method=METHOD)

# Cauchy distribution has Zipf tails...
DESIRED_ALPHA = 2.0


def test_cauchy_estimation(size=0.01):
    """Test the estimation of the Levy-Stable scaling exponent."""
    utilities.test_scaling_exponent_estimation(DESIRED_ALPHA, RESULT1, size)
    utilities.test_scaling_exponent_estimation(DESIRED_ALPHA, RESULT2, size)


def test_cauchy_goodness_of_fit(size=0.05):
    """Test the goodness of fit of the Pareto distribution."""
    test_kwargs = {'seed': None, 'result': RESULT1, 'data': CAUCHY_POSITIVE_TAIL,
                   'scale': None, 'quantile': QUANTILE, 'method': METHOD}
    pvalue, _ = distributions.Pareto.test_goodness_of_fit(**test_kwargs)
    assert pvalue > size, "Goodness of fit test Type I error!"

    test_kwargs = {'seed': None, 'result': RESULT2, 'data': CAUCHY_NEGATIVE_TAIL,
                   'scale': None, 'quantile': QUANTILE, 'method': METHOD}
    pvalue, _ = distributions.Pareto.test_goodness_of_fit(**test_kwargs)
    assert pvalue > size, "Goodness of fit test Type I error!"
