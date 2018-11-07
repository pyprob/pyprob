__version__ = '0.13.dev1'

from .util import TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, ObserveEmbedding, set_verbosity, set_random_seed, set_cuda
from .state import sample, observe, tag
from .address_dictionary import AddressDictionary
from .model import Model, ModelRemote
from .diagnostics import Diagnostics
