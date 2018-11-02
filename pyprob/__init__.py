__version__ = '0.12.dev1'

from .util import TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, ObserveEmbedding, set_verbosity, set_random_seed, set_cuda
from .state import sample, observe, tag
from .model import Model, ModelRemote
from .diagnostics import Diagnostics
