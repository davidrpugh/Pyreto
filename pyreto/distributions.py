import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats
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

    @staticmethod
    def _quantile(x, xmin, *args, **kwargs):
        """Inverse of the cumulative distibution function (CDF)."""
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
        C = cls._normalization_constant(xmin, *args, **params)
        f = cls._density_function(x, *args, **params)
        return C * f

    @classmethod
    def _rvs(cls, random_state, N, xmin, *args, **kwargs):
        sample = stats.uniform.rvs(0, 1, size=N, random_state=random_state)
        return cls._quantile(sample, xmin, *args, **kwargs)

    @classmethod
    def _generate_synthetic_data(cls, prng, clean_data, result):
        N = clean_data.count()
        body = clean_data[clean_data < result.xmin]
        body_prob = body.count() / N
        successes = stats.binom.rvs(1, body_prob, size=N, random_state=prng)
        n_body = np.sum(successes)
        body_sample = prng.choice(body, n_body)
        n_tail = N - n_body
        tail_sample = cls._rvs(prng, n_tail, result.xmin, **result.params)
        synthetic_data = pd.Series(np.hstack((body_sample, tail_sample)),
                                   index=clean_data.index)
        return synthetic_data


class Exponential(Distribution):

    @classmethod
    def fit(cls, data, xmin):
        """Fit Exponential distribution to data using maximum likelihood."""
        cleaned_data = cls._clean_data(data)
        tail_data = cleaned_data[cleaned_data >= xmin]
        gamma_hat = 1 / np.mean(tail_data - xmin)

        # create the FitResult object...
        n_tail = tail_data.count()
        gamma_se = gamma_hat / n_tail
        log_likelihood = cls._log_likelihood(tail_data.values, xmin, gamma_hat)
        result_kwargs = {'params': {'gamma': gamma_hat}, 'n_tail': n_tail,
                         'standard_errors': {'gamma': gamma_se},
                         'log_likelihood': log_likelihood, 'D': None}
        result = results.FitResult(**result_kwargs)

        return result

    @classmethod
    def _cdf(cls, x, xmin, gamma):
        """Cumulative distribution function for Exponential distribution."""
        return np.exp(gamma * xmin) * (np.exp(-gamma * xmin) - np.exp(-gamma * x))

    @staticmethod
    def _density_function(x, gamma):
        """Density function for the Exponential distribution."""
        return np.exp(-gamma * x)

    @staticmethod
    def _normalization_constant(xmin, gamma):
        """Normalization constant for the Exponential distribution."""
        return gamma * np.exp(gamma * xmin)


class LogNormal(Distribution):

    @classmethod
    def fit(cls, data, xmin, initial_guess, method, solver_options):
        result = optimize.minimize(cls._objective, initial_guess, (xmin, data),
                                   method, **solver_options)
        return result

    @classmethod
    def _cdf(cls, x, xmin, mu, sigma):
        return integrate.quad(cls._pdf, xmin, x, args=(xmin, mu, sigma))

    @staticmethod
    def _density_function(x, mu, sigma):
        return (1 / x) * np.exp(-((np.log(x) - mu)**2 / (2 * sigma**2)))

    @staticmethod
    def _normalization_constant(xmin, mu, sigma):
        return (2 / (np.pi * sigma**2))**0.5 * (special.erfc((np.log(xmin) - mu) / (2**0.5 * sigma)))**-1


class Pareto(Distribution):

    @classmethod
    def fit(cls, data, xmin, quantile=0.95, discrete=False,
            approx=False, method='bounded', solver_opts=None):
        clean_data = cls._clean_data(data)
        if xmin is None:
            solver_opts = {} if solver_opts is None else solver_opts
            idxs = clean_data < clean_data.quantile(quantile)
            candidate_xmins = clean_data[idxs].unique()
            xmin, D = cls._find_optimal_xmin(candidate_xmins, clean_data,
                                             discrete, approx, method,
                                             solver_opts)
        else:
            D = cls._compute_ks_distance(xmin, clean_data, discrete, approx)
        alpha_hat, tail_data = cls._fit_maximum_likelihood(xmin, clean_data,
                                                           discrete, approx)

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

    @classmethod
    def test_goodness_of_fit(cls, seed, result, data, xmin=None,
                             quantile=0.99, discrete=False, approx=False,
                             method='brute', solver_opts=None,
                             replications=1000):
        prng = np.random.RandomState(seed)
        ks_distances = np.empty(replications)
        clean_data = cls._clean_data(data)
        for i in range(replications):
            tmp_data = cls._generate_synthetic_data(prng, clean_data, result)
            tmp_result = cls.fit(tmp_data, xmin, quantile, discrete, approx,
                                 method)
            ks_distances[i] = tmp_result.D
        pvalue = ks_distances[ks_distances > result.D].mean()
        return pvalue, ks_distances

    @staticmethod
    def _cdf(x, xmin, alpha):
        """Cumulative distribution function (CDF)."""
        return 1 - (xmin / x)**(alpha - 1)

    @staticmethod
    def _density_function(x, alpha):
        return x**-alpha

    @classmethod
    def _fit_maximum_likelihood(cls, xmin, clean_data, discrete, approx):
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
            alpha_hat, tail_data = cls._mle_discrete(xmin, clean_data, approx)
        else:
            alpha_hat, tail_data = cls._mle_continuous(xmin, clean_data)
        return alpha_hat, tail_data

    @staticmethod
    def _mle_continuous(xmin, clean_data):
        """Maximum likelihood estimator of the scaling exponent."""
        tail_data = clean_data[clean_data >= xmin]
        n = tail_data.count()
        alpha_hat = 1 + n * (np.log(tail_data / xmin).sum())**-1
        return alpha_hat, tail_data

    @staticmethod
    def _mle_discrete(xmin, clean_data, approx):
        """Maximum likelihood estimator of the scaling exponent."""
        tail_data = clean_data[clean_data >= xmin]
        n = tail_data.count()
        if approx:
            alpha_hat = 1 + n * (np.log(tail_data / (xmin - 0.5)).sum())**-1
        else:
            raise NotImplementedError
        return alpha_hat, tail_data

    @staticmethod
    def _normalization_constant(xmin, alpha):
        return (alpha - 1) * xmin**(alpha - 1)

    @staticmethod
    def _quantile(q, xmin, alpha):
        """Inverse of the cumulative distibution function (CDF)."""
        return xmin / (1 - q)**(1 / (alpha - 1))

    @classmethod
    def _brute_force_minimize(cls, xmins, clean_data, discrete, approx):
        Ds = [cls._compute_ks_distance(xmin, clean_data, discrete, approx) for xmin in xmins]
        idx = np.argmin(Ds)
        return xmins[idx], Ds[idx]

    @classmethod
    def _compute_ks_distance(cls, xmin, clean_data, discrete, approx):
        """Compute the Kolmogorov-Smirnov (KS) distance."""
        alpha_hat, tail_data = cls._fit_maximum_likelihood(xmin, clean_data,
                                                           discrete, approx)
        ecdf = cls._ecdf(tail_data)
        cdf = cls._cdf(ecdf.x[1:], xmin, alpha_hat)
        D = np.max(np.abs(ecdf.y[1:] - cdf))
        return D

    @classmethod
    def _find_optimal_xmin(cls, xmins, clean_data, discrete, approx, method,
                           solver_opts):
        """Find optimal xmin by minimizing Kolmogorov-Smirnov (KS) distance."""
        if method == 'brute':
            xmin, D = cls._brute_force_minimize(xmins, clean_data, discrete,
                                                approx)
        elif method == 'bounded':
            result = optimize.fminbound(cls._compute_ks_distance,
                                        xmins.min(),
                                        xmins.max(),
                                        args=(clean_data, discrete, approx),
                                        full_output=True,
                                        **solver_opts)
            xmin, D, _, _ = result
        else:
            raise ValueError
        return xmin, D


class StretchedExponential(Distribution):

    @classmethod
    def fit(cls, data, xmin):
        """
        Fit Stretched Exponential distribution to data using maximum
        likelihood.

        """
        raise NotImplementedError

    @classmethod
    def _cdf(cls, x, xmin, beta, gamma):
        raise NotImplementedError

    @staticmethod
    def _density_function(x, beta, gamma):
        return x**(beta - 1) * np.exp(-gamma * x**beta)

    @staticmethod
    def _normalization_constant(xmin, beta, gamma):
        return beta * gamma * np.exp(gamma * xmin**beta)
