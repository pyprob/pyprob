__version__ = '1.0.1a'

from .util import TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, ImportanceWeighting, Optimizer, LearningRateScheduler, ObserveEmbedding, set_verbosity, set_random_seed, set_device
from .state import sample, observe, tag
from .address_dictionary import AddressDictionary
from .model import Model, RemoteModel
