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
    IMPORTANCE_SAMPLING_WITH_PRIOR = 2  # Record traces, importance sampling with prior
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 3  # Record traces, importance sampling with proposals from inference network (inference compilation inference)
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK_TRAIN = 4  # Record traces, training data generation for inference network, interpret 'observe' as 'sample' (inference compilation training)
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 5  # Record traces for single-site Metropolis Hastings sampling, http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf and https://arxiv.org/abs/1507.00996


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
