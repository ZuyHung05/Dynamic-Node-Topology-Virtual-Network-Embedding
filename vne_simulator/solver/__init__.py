import itertools
from .solver import Solver
from .heuristic import *
from .meta_heuristic import *
from .registry import REGISTRY, register, get

from . import heuristic, meta_heuristic

try:
    from .learning import *
    from . import learning
except ImportError:
    learning = None


SOLVERS = {
    'heuristic': tuple(heuristic.__all__),
    'meta_heuristic': tuple(meta_heuristic.__all__),
    'learning': tuple(getattr(learning, '__all__', ())),
}
SOLVERS['all'] = tuple(itertools.chain.from_iterable(SOLVERS.values()))

__all__ = list(SOLVERS['all']) + [
    'Solver',
    'REGISTRY',
    'register',
    'get',
]
