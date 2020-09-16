__version__ = '1.3.0.dev4'

from .util import TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, Optimizer, LearningRateScheduler, ObserveEmbedding, set_verbosity, set_device, seed
from .state import sample, observe, factor, tag
from .address_dictionary import AddressDictionary
from .model import Model, RemoteModel
