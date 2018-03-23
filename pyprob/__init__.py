__version__ = '0.10.0.dev11'

import enum


class ObserveEmbedding(enum.Enum):
    FULLY_CONNECTED = 0
    CONVNET_2D_5C = 1
    CONVNET_3D_4C = 2


class SampleEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class TraceMode(enum.Enum):
    NONE = 0  # No trace recording, forward sample
    RECORD = 1  # Record traces
    RECORD_IMPORTANCE = 2  # Record traces, importance sampling with prior
    RECORD_TRAIN_INFERENCE_NETWORK = 3  # Record traces, training data generation for inference network, interpret 'observe' as 'sample' (inference compilation training)
    RECORD_USE_INFERENCE_NETWORK = 4  # Record traces, importance sampling with proposals from inference network (inference compilation inference)


class InferenceEngine(enum.Enum):
    IMPORTANCE_SAMPLING = 0  # Type: IS
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 1  # Type: IS
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 2  # Type: MCMC


del enum
from .util import set_random_seed, set_cuda, set_verbosity
from .model import Model, ModelRemote
from .state import sample, observe


# import time
# set_random_seed(int(time.time()))
# set_cuda(True)

del util
del model
del state
