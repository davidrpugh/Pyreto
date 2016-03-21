"""
Objects imported here will live in the `pyreto` namespace

"""
from pkg_resources import get_distribution

from . import distributions
from . import pyreto
from . import testing


__all__ = ["distributions", "pyreto", 'testing']
__version__ = get_distribution('Pyreto').version
