import collections


FitResult = collections.namedtuple('FitResult', ['params', 'standard_errors', 'xmin', 'D', 'n_tail', 'log_likelihood'])
