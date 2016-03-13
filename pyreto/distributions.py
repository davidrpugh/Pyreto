import numpy as np
from scipy import optimize
import statsmodels as sm

from . import results


class Distribution:

    @staticmethod
    def _clean_data(series):
        """Remove any non-positive, null, or NAN observations."""
        return series[series > 0].dropna().sort_values()

    @staticmethod
    def _density_function(x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _ecdf(series):
        """Empirical cumulative distribution function (ECDF)."""
        return sm.distributions.ECDF(series)

    @staticmethod
    def _normalization_constant(xmin, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _log_likelihood(cls, x, xmin, *args, **kwargs):
        """Pointwise log-likelihood function."""
        return np.log(cls._pdf(x, xmin, *args, **kwargs))

    @classmethod
    def _mle_objective(cls, params, xmin, series, **kwargs):
        """Objective function for maximum likelihood estimation.."""
        log_likelihood = cls._log_likelihood(series, xmin, *params, **kwargs)
        return -log_likelihood.sum()

    @classmethod
    def _pdf(cls, x, xmin, *args, **params):
        """Probability density function (pdf)."""
        C = cls._normalization_constant(x, *args, **params)
        f = cls._density_function(xmin, *args, **params)
        return C * f


class Pareto(Distribution):

    @classmethod
    def fit(cls, data, xmin=None, quantile=0.95, method='bounded',
            solver_opts=None):
        clean_data = cls._clean_data(data)
        if xmin is None:
            solver_opts = {} if solver_opts is None else solver_opts
            idxs = clean_data < clean_data.quantile(quantile)
            candidate_xmins = clean_data[idxs]
            xmin, D = cls._find_optimal_xmin(candidate_xmins,
                                             clean_data,
                                             method,
                                             solver_opts)
        else:
            D = cls._compute_ks_distance(xmin, clean_data)
        alpha_hat, tail_data = cls._fit_maximum_likelihood(xmin, clean_data)

        # create the FitResult object...
        n_tail = tail_data.count()
        alpha_se = (alpha_hat - 1) / n_tail**0.5
        log_likelihood = cls._log_likelihood(tail_data.values, xmin, alpha_hat)
        fit_result_kwargs = {'params': {'alpha': alpha_hat}, 'xmin': xmin,
                             'D': D, 'n_tail': n_tail,
                             'standard_errors': {'alpha': alpha_se},
                             'log_likelihood': log_likelihood}
        result = results.FitResult(**fit_result_kwargs)

        return result

    @staticmethod
    def _cdf(x, xmin, alpha):
        """Cumulative distribution function (CDF)."""
        return 1 - (xmin / x)**(alpha - 1)

    @staticmethod
    def _density_function(x, alpha):
        return x**-alpha

    @staticmethod
    def _fit_maximum_likelihood(xmin, clean_data):
        """Maximum likelihood estimator of the scaling exponent."""
        tail_data = clean_data[clean_data >= xmin]
        n = tail_data.count()
        alpha_hat = 1 + n * (np.log(tail_data / xmin).sum())**-1
        return alpha_hat, tail_data

    @staticmethod
    def _normalization_constant(xmin, alpha):
        return (alpha - 1) * xmin**(alpha - 1)

    @classmethod
    def _brute_force_minimize(cls, xmins, clean_data):
        Ds = [cls._compute_ks_distance(xmin, clean_data) for xmin in xmins]
        idx = np.argmin(Ds)
        return xmins.values[idx], Ds[idx]

    @classmethod
    def _compute_ks_distance(cls, xmin, clean_data):
        """Compute the Kolmogorov-Smirnov (KS) distance."""
        alpha_hat, tail_data = cls._fit_maximum_likelihood(xmin, clean_data)
        ecdf = cls._ecdf(tail_data)
        cdf = cls._cdf(ecdf.x[1:], alpha_hat, xmin)
        D = np.max(np.abs(ecdf.y[1:] - cdf))
        return D

    @classmethod
    def _find_optimal_xmin(cls, xmins, clean_data, method, solver_opts):
        """Find optimal xmin by minimizing Kolmogorov-Smirnov (KS) distance."""
        if method == 'brute':
            xmin, D = cls._brute_force_minimize(xmins, clean_data)
        elif method == 'bounded':
            result = optimize.fminbound(cls._compute_ks_distance,
                                        xmins.min(),
                                        xmins.max(),
                                        args=(clean_data,),
                                        full_output=True,
                                        **solver_opts)
            xmin, D, _, _ = result
        else:
            raise ValueError
        return xmin, D
