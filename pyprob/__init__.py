__version__ = '0.10.0'

from .util import ObserveEmbedding, SampleEmbedding, TraceMode, InferenceEngine, InferenceNetwork, PriorInflation, Optimizer, TrainingObservation, set_random_seed, set_cuda, set_verbosity
from .model import Model, ModelRemote
from .state import sample, observe


# import time
# set_random_seed(int(time.time()))
# set_cuda(True)

del util
del model
del state
