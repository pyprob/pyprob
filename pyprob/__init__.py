__version__ = '0.13.3.dev2'

from .util import TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, Optimizer, LearningRateScheduler, ObserveEmbedding, set_verbosity, set_random_seed, set_device
from .state import sample, observe, tag
from .address_dictionary import AddressDictionary
from .model import Model, RemoteModel
