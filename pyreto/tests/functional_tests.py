import numpy as np
import pandas as pd

from pyreto import distributions


def test_fire_sizes(desired_alpha=2.2, desired_xmin=6234, decimal=1):
    """Replicate Clauset et al (2009) analysis of forest fires."""
    data_url = "http://tuvalu.santafe.edu/~aaronc/powerlaws/data/fires.txt"
    fire_size = pd.read_csv(data_url, names=['acres'])

    # check that I get same estimate for alpha given reported xmin...
    result1 = distributions.Pareto.fit(fire_size.acres, xmin=desired_xmin)
    np.testing.assert_almost_equal(result1.params['alpha'], desired_alpha,
                                   decimal=decimal)

    # check that I get the same estimates for both alpha and xmin...
    result2 = distributions.Pareto.fit(fire_size.acres, xmin=None,
                                       quantile=0.999, method='brute')
    np.testing.assert_almost_equal(result2.params['alpha'], desired_alpha,
                                   decimal=decimal)
    np.testing.assert_almost_equal(result2.xmin, desired_xmin, decimal=decimal)
