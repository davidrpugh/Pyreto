import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels as sm

from . import results


class Distribution:

    _distribution = None

    @staticmethod
    def _clean_data(series):
        """Remove any non-positive, null, or NAN observations."""
        return series[series > 0].dropna().sort_values()

    @staticmethod
    def _split_data(data, xmin):
        """Split a data series into body and tail."""
        return data[data <= xmin], data[data > xmin]

    @classmethod
    def cdf(cls, x, *args, **kwargs):
        """Cumulative distribution function (CDF)."""
        return cls._distribution.cdf(x, *args, **kwargs)

    @staticmethod
    def ecdf(series):
        """Empirical cumulative distribution function (ECDF)."""
        return sm.distributions.ECDF(series).y[1:]

    @classmethod
    def logpdf(cls, x, *args, **kwargs):
        """Pointwise log-likelihood function."""
        return cls._distribution.logpdf(x, *args, **kwargs)

    @classmethod
    def pdf(cls, x, *args, **kwargs):
        """Probability density function (pdf)."""
        return cls._distribution.pdf(x, *args, **kwargs)

    @classmethod
    def rvs(cls, *args, **kwargs):
        """Generate random samples from the underlying distribution."""
        return cls._distribution.rvs(*args, **kwargs)


class Exponential(Distribution):

    _distribution = stats.expon

    @classmethod
    def fit(cls, data, floc, *args, **kwargs):
        """Fit Exponential distribution to data using maximum likelihood."""
        _, tail = cls._split_data(data, floc)
        lambda_hat = 1 / np.mean(tail - floc)

        # create the FitResult object...
        n_tail = tail.count()
        log_likelihood = cls.logpdf(tail, loc=floc, scale=1 / lambda_hat)
        result_kwargs = {'params': {'loc': floc, 'scale': 1 / lambda_hat},
                         'n_tail': n_tail, 'standard_errors': {'loc': None, 'scale': None},
                         'log_likelihood': log_likelihood, 'D': None}
        result = results.FitResult(**result_kwargs)

        return result


class LogNormal(Distribution):

    _distribution = stats.lognorm

    @classmethod
    def fit(cls, data, floc, *args, **kwargs):
        _, tail = cls._split_data(data, floc)
        s, loc, scale = cls._distribution.fit(data, *args, floc=floc, **kwargs)

        # create the FitResult object...
        n_tail = tail.count()
        log_likelihood = cls.logpdf(tail, s, loc, scale)
        result_kwargs = {'params': {'s': s, 'loc': loc, 'scale': scale},
                         'n_tail': n_tail, 'standard_errors': {},
                         'log_likelihood': log_likelihood, 'D': None}
        result = results.FitResult(**result_kwargs)

        return result


class Pareto(Distribution):

    _distribution = stats.pareto

    @classmethod
    def fit(cls, data, loc=0, scale=None, quantile=0.95, discrete=False,
            approx=False, normalize=False, method='bounded', solver_opts=None):
        clean_data = cls._clean_data(data)
        if scale is None:
            solver_opts = {} if solver_opts is None else solver_opts
            idxs = clean_data < clean_data.quantile(quantile)
            candidate_xmins = clean_data[idxs].unique()
            kwargs = {'xmins': candidate_xmins, 'loc': loc,
                      'data': clean_data, 'discrete': discrete,
                      'approx': approx, 'method': method,
                      'normalize': normalize, 'solver_opts': solver_opts}
            scale, D = cls._find_optimal_xmin(**kwargs)
        else:
            kwargs = {'data': clean_data, 'loc': loc, 'scale': scale,
                      'discrete': discrete, 'approx': approx,
                      'normalize': normalize}
            D = cls._ks_objective(**kwargs)

        _, tail = cls._split_data(data, scale)
        kwargs = {'data': tail, 'loc': loc, 'scale': scale,
                  'discrete': discrete, 'approx': approx}
        b_hat = cls._fit_maximum_likelihood(**kwargs)

        # create the FitResult object...
        n_tail = tail.count()
        b_se = b_hat / n_tail**0.5  # classic MLE standard errors!
        log_likelihood = cls.logpdf(tail, b_hat, loc, scale)
        kwargs = {'params': {'b': b_hat, 'loc': loc, 'scale': scale}, 'D': D,
                  'n_tail': n_tail, 'standard_errors': {'b': b_se},
                  'log_likelihood': log_likelihood}
        result = results.FitResult(**kwargs)

        return result

    @classmethod
    def test_goodness_of_fit(cls, seed, result, data, loc=0, scale=None,
                             quantile=0.99, discrete=False, approx=False,
                             method='brute', solver_opts=None,
                             replications=1000):
        prng = np.random.RandomState(seed)
        ks_distances = np.empty(replications)
        clean_data = cls._clean_data(data)
        for i in range(replications):
            tmp_data = cls._generate_synthetic_data(prng, clean_data, result)
            tmp_kwargs = {'data': tmp_data, 'loc': loc, 'scale': scale,
                          'quantile': quantile, 'discrete': discrete,
                          'approx': approx, 'method': method,
                          'solver_opts': solver_opts}
            tmp_result = cls.fit(**tmp_kwargs)
            ks_distances[i] = tmp_result.D
        pvalue = ks_distances[ks_distances > result.D].mean()
        return pvalue, ks_distances

    @classmethod
    def _brute_force_minimize(cls, xmins, loc, data, discrete, approx, normalize):
        """Brute force minimization of the KS distance."""
        kwargs = {'loc': loc, 'data': data, 'discrete': discrete,
                  'approx': approx, 'normalize': normalize}
        Ds = [cls._ks_objective(xmin, **kwargs) for xmin in xmins]
        idx = np.argmin(Ds)
        return xmins[idx], Ds[idx]

    @staticmethod
    def _compute_ks_distance(ecdf, cdf, normalize):
        """Compute the Kolmogorov-Smirnov (KS) distance."""
        difference = np.abs(ecdf - cdf)
        if normalize:
            difference /= (cdf * (1 - cdf))
        D = np.max(np.abs(difference))
        return D

    @classmethod
    def _find_optimal_xmin(cls, xmins, loc, data, discrete, approx, normalize,
                           method, solver_opts):
        """Find optimal xmin by minimizing Kolmogorov-Smirnov (KS) distance."""
        if method == 'brute':
            kwargs = {'loc': loc, 'data': data, 'discrete': discrete,
                      'approx': approx, 'normalize': normalize}
            xmin, D = cls._brute_force_minimize(xmins, **kwargs)
        elif method == 'bounded':
            args = (loc, data, discrete, approx, normalize)
            result = optimize.fminbound(cls._ks_objective, xmins.min(),
                                        xmins.max(), args=args,
                                        full_output=True, **solver_opts)
            xmin, D, _, _ = result
        else:
            raise ValueError
        return xmin, D

    @classmethod
    def _fit_maximum_likelihood(cls, data, loc, scale, discrete, approx):
        r"""
        Fit a Pareto distribution to some data using maximum likelihood.

        Notes
        -----
        For a given value of $x_{min}$, the maximum likelihood estimator for
        the scaling exponent is
        \begin{equation}\label{eq:plawMLE}
            \hat{\alpha} = 1 +  n\left[\sum_{i=1}^{n}\mathrm{ln}\ \left(\frac{x_{i}}{x_{min}}\right)\right]^{-1}.
        \end{equation}
        Equation \ref{eq:plawMLE}, is equivalent to the \cite{hill1975simple}
        estimator, and has been shown to be asymptotically normal
        \cite{hall1982some} and consistent \cite{mason1982laws}. The standard
        error of $\hat{\alpha}$ is
        \begin{equation}\label{eq:seplawMLE}
            \sigma = \frac{\hat{\alpha} - 1}{\sqrt{n}} + \mathcal{O}\left(n^{-1}\right)
        \end{equation}
        """
        if discrete:
            b_hat = cls._mle_discrete(data, loc, scale, approx)
        else:
            b_hat = cls._mle_continuous(data, loc, scale)
        return b_hat

    @classmethod
    def _generate_synthetic_data(cls, prng, data, result):
        """Generate synthetic data for goodness of fit test."""
        N = data.count()
        body, _ = cls._split_data(data, result.params['scale'])
        successes = stats.binom.rvs(1, body.count() / N, size=N, random_state=prng)
        body_sample = prng.choice(body, successes.sum(), replace=True)
        n_tail = N - body_sample.size
        tail_sample = cls.rvs(size=n_tail, random_state=prng, **result.params)
        full_sample = np.hstack((body_sample, tail_sample))
        synthetic_data = pd.Series(full_sample, index=data.index)
        return synthetic_data

    @classmethod
    def _ks_objective(cls, scale, loc, data, discrete, approx, normalize):
        """
        Return the Kolmogorov-Smirnov (KS) distance between ECDF of the tail
        data and the theoretical CDF computed using the estimated scaling
        exponent for given loc and scale parameters.

        """
        # estimate the scaling exponent given values for loc and scale
        _, tail = cls._split_data(data, scale)
        fit_kwargs = {'data': tail, 'loc': loc, 'scale': scale,
                      'discrete': discrete, 'approx': approx}
        b_hat = cls._fit_maximum_likelihood(**fit_kwargs)

        # compute the KS distance...
        _, tail = cls._split_data(data, scale)
        ecdf = cls.ecdf(tail)
        cdf = cls.cdf(tail, b_hat, loc, scale)
        D = cls._compute_ks_distance(ecdf, cdf, normalize)

        return D

    @staticmethod
    def _mle_continuous(data, loc, scale):
        """Maximum likelihood estimator of the scaling exponent."""
        b_hat = ((np.log(data - loc) - np.log(scale)).mean())**-1
        return b_hat

    @staticmethod
    def _mle_discrete(data, loc, scale, approx):
        """Maximum likelihood estimator of the scaling exponent."""
        if approx:
            b_hat = (np.log(data / (scale - 0.5)).mean())**-1
        else:
            raise NotImplementedError
        return b_hat


class StretchedExponential(Distribution):

    _distribution = stats.exponweib

    @classmethod
    def fit(cls, data, floc):
        """
        Fit Stretched Exponential distribution to data using maximum
        likelihood.

        """
        _, tail = cls._split_data(data, floc)
        a, c, loc, scale = cls._distribution.fit(data, fa=1, floc=floc)

        # create the FitResult object...
        n_tail = tail.count()
        log_likelihood = cls.logpdf(tail, a, c, loc, scale)
        standard_errors = {'a': None, 'c': None, 'loc': None, 'scale': None}
        result_kwargs = {'params': {'a': a, 'c': c, 'loc': loc, 'scale': scale},
                         'n_tail': n_tail, 'standard_errors': standard_errors,
                         'log_likelihood': log_likelihood, 'D': None}
        result = results.FitResult(**result_kwargs)

        return result
