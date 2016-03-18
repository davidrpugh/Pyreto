class FitResult:

    def __init__(self, params, standard_errors, D, n_tail, log_likelihood):
        self._params = params
        self._standard_errors = standard_errors
        self._D = D
        self._n_tail = n_tail
        self._log_likelihood = log_likelihood

    @property
    def params(self):
        return self._params

    @property
    def standard_errors(self):
        return self._standard_errors

    @property
    def D(self):
        return self._D

    @property
    def n_tail(self):
        return self._n_tail

    @property
    def log_likelihood(self):
        return self._log_likelihood
