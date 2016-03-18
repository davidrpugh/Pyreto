import numpy as np
import pandas as pd

from .. import distributions


def test_fire_sizes_estimation(desired_alpha=2.2, desired_xmin=6324, decimal=1):
    """Replicate Clauset et al (2009) analysis of forest fires."""
    data_url = "http://tuvalu.santafe.edu/~aaronc/powerlaws/data/fires.txt"
    fire_size = pd.read_csv(data_url, names=['acres'])

    # check that I get same estimate for alpha given reported xmin...
    result1 = distributions.Pareto.fit(fire_size.acres, scale=desired_xmin)
    actual_alpha = result1.params['b'] + 1
    np.testing.assert_almost_equal(actual_alpha, desired_alpha, decimal)

    # check that I get the same estimates for both alpha and xmin...
    result2 = distributions.Pareto.fit(fire_size.acres, scale=None,
                                       quantile=0.999, method='brute')
    actual_alpha, actual_xmin = result2.params['b'] + 1, result2.params['scale']
    np.testing.assert_almost_equal(actual_alpha, desired_alpha, decimal)
    np.testing.assert_almost_equal(actual_xmin, desired_xmin, decimal)
