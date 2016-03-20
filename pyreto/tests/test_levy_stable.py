import pandas as pd
from scipy import stats

from .. import distributions
from . import utilities

# Fit the pareto distribution to Levy-Stable data
DESIRED_ALPHA = stats.uniform.rvs(1, 3)
BETA = 1.0  # forces rvs to be strictly positive
STABLE_RVS = stats.levy_stable.rvs(DESIRED_ALPHA - 1, BETA, size=1000)
STABLE_DATA = pd.Series(STABLE_RVS, name='samples')
QUANTILE = 0.99
METHOD = 'bounded'
RESULT = distributions.Pareto.fit(STABLE_DATA, scale=None, quantile=QUANTILE,
                                  method=METHOD)


def test_levy_stable_estimation(size=0.01):
    """Test the estimation of the Levy-Stable scaling exponent."""
    utilities.test_scaling_exponent_estimation(DESIRED_ALPHA, RESULT, size)


def test_levy_stable_goodness_of_fit(size=0.05):
    """Test the goodness of fit of the Pareto distribution."""
    test_kwargs = {'seed': None, 'result': RESULT, 'data': STABLE_DATA,
                   'scale': None, 'quantile': QUANTILE, 'method': METHOD}
    pvalue, _ = distributions.Pareto.test_goodness_of_fit(**test_kwargs)
    assert pvalue > size, "Goodness of fit test Type I error!"
