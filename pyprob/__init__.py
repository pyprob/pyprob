__version__ = '0.10.0.dev12'

from .util import ObserveEmbedding, SampleEmbedding, TraceMode, InferenceEngine, Optimizer, InferenceNetworkTrainingMode, set_random_seed, set_cuda, set_verbosity, set_inference_network_training_mode
from .model import Model, ModelRemote
from .state import sample, observe


# import time
# set_random_seed(int(time.time()))
# set_cuda(True)

del util
del model
del state
