"""
Objects imported here will live in the `pyreto` namespace

"""
from pkg_resources import get_distribution

from . import distributions
from . import pyreto

__all__ = ["distributions", "pyreto"]
__version__ = get_distribution('Pyreto').version
