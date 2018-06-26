import sys
import torch
from torch.autograd import Variable
import numpy as np
import random
import datetime
import inspect
import time
import math
from termcolor import colored
import enum

import matplotlib
matplotlib.use('Agg') # Do not use X server
import matplotlib.pyplot as plt

from .distributions import Empirical, Categorical


_random_seed = 0
_epsilon = 1e-8
_log_epsilon = math.log(_epsilon)
_print_refresh_rate = 0.2  # seconds


class ObserveEmbedding(enum.Enum):
    FULLY_CONNECTED = 0
    CONVNET_2D_5C = 1
    CONVNET_3D_4C = 2


class SampleEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class TraceMode(enum.Enum):
    NONE = 0  # No trace recording, forward sample trace results only
    PRIOR = 1
    POSTERIOR = 2


class InferenceEngine(enum.Enum):
    IMPORTANCE_SAMPLING = 0  # Type: IS; Importance sampling with proposals from prior
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 1  # Type: IS; Importance sampling with proposals from inference network
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 2  # Type: MCMC; Lightweight (single-site) Metropolis Hastings sampling, http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf and https://arxiv.org/abs/1507.00996
    RANDOM_WALK_METROPOLIS_HASTINGS = 3  # Type: MCMC; Lightweight Metropolis Hastings with single-site proposal kernels that depend on the value of the site


class InferenceNetwork(enum.Enum):
    SIMPLE = 0  # A simple inference network that maps an observation embedding to proposals specializing in each address
    LSTM = 1  # An advanced LSTM-based inference network that maintains execution state, keeps track of sampled values, learns address embeddings


class Optimizer(enum.Enum):
    ADAM = 0
    SGD = 1


class TrainingObservation(enum.Enum):
    OBSERVE_DIST_SAMPLE = 0
    OBSERVE_DIST_MEAN = 1


class PriorInflation(enum.Enum):
    DISABLED = 0
    ENABLED = 1


def set_random_seed(seed=123):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


set_random_seed()

Tensor = torch.FloatTensor
_cuda_enabled = False
_cuda_device = 0


def set_cuda(enabled, device=0):
    global Tensor
    global _cuda_enabled
    global _cuda_device
    if torch.cuda.is_available() and enabled:
        _cuda_enabled = True
        _cuda_device = device
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        Tensor = torch.cuda.FloatTensor
    else:
        _cuda_enabled = False
        Tensor = torch.FloatTensor


verbosity = 2


def set_verbosity(v=2):
    global verbosity
    verbosity = v


def to_variable(value, *args, **kwargs):
    ret = None
    if isinstance(value, Variable):
        ret = value
    elif torch.is_tensor(value):
        ret = Variable(value.float(), *args, **kwargs)
    elif isinstance(value, np.ndarray):
        ret = Variable(torch.from_numpy(value.astype(float)).float(), *args, **kwargs)
    elif value is None:
        ret = None
    elif isinstance(value, (list, tuple)):
        if isinstance(value[0], Variable):
            ret = torch.stack(value).float()
        elif torch.is_tensor(value[0]):
            ret = torch.stack(list(map(lambda x: Variable(x, *args, **kwargs), value))).float()
        elif (type(value[0]) is float) or (type(value[0]) is int):
            ret = torch.stack(list(map(lambda x: Variable(Tensor([x]), *args, **kwargs), value))).float().view(-1)
        else:
            ret = Variable(Tensor(value))
    else:
        ret = Variable(Tensor([float(value)]), *args, **kwargs)
    if _cuda_enabled:
        return ret.float().cuda()
    else:
        return ret.float()


def to_numpy(value):
    if isinstance(value, Variable):
        return value.data.cpu().numpy()
    elif torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except:
            raise TypeError('Cannot convert to Numpy array.')


def log_sum_exp(tensor, keepdim=True):
    r"""
    Numerically stable implementation for the `LogSumExp` operation. The
    summing is done along the last dimension.
    Args:
        tensor (torch.Tensor or torch.autograd.Variable)
        keepdim (Boolean): Whether to retain the last dimension on summing.
    """
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()


def fast_np_random_choice(values, probs_cumsum):
    # See https://mobile.twitter.com/RadimRehurek/status/928671225861296128
    return values[min(np.searchsorted(probs_cumsum, random.random()), len(values) - 1)]


def debug(*expressions):
    print('\n\n' + colored(inspect.stack()[1][3], 'white', attrs=['bold']))
    frame = sys._getframe(1)
    max_str_length = 0
    for expression in expressions:
        if len(expression) > max_str_length:
            max_str_length = len(expression)
    for expression in expressions:
        val = eval(expression, frame.f_globals, frame.f_locals)
        if isinstance(val, np.ndarray):
            val = val.tolist()
        print('  {} = {}'.format(expression.ljust(max_str_length), repr(val)))


def pack_observes_to_variable(observes):
    try:
        return torch.stack([to_variable(o) for o in observes])
    except:
        try:
            return torch.cat([to_variable(o).view(-1) for o in observes])
        except:
            try:
                return to_variable(observes)
            except:
                return to_variable(Tensor())


def one_hot(dim, i):
    t = Tensor(dim).zero_()
    t.narrow(0, i, 1).fill_(1)
    return to_variable(t)


def kl_divergence_normal(p, q):
    return torch.log(q.stddev) - torch.log(p.stddev) + (p.stddev.pow(2) + (p.mean - q.mean).pow(2)) / (2 * q.stddev.pow(2)) - 0.5


def kl_divergence_categorical(p, q):
    return safe_torch_sum(clamp_prob(p._probs) * torch.log(clamp_prob(p._probs) / clamp_prob(q._probs)))


def empirical_to_categorical(empirical_dist, max_val=None):
    empirical_dist = Empirical(empirical_dist.values, clamp_log_prob(torch.log(empirical_dist.weights)), combine_duplicates=True).map(int)
    if max_val is None:
        max_val = int(empirical_dist.max)
    probs = Tensor(max_val + 1).zero_()
    for i in range(empirical_dist.length):
        val = empirical_dist.values[i]
        if val <= max_val:
            probs[val] = float(empirical_dist.weights[i])
    return Categorical(probs)


def has_nan_or_inf(value):
    if isinstance(value, Variable):
        value = value.data
    if torch.is_tensor(value):
        value = np.sum(value.cpu().numpy())
        return np.isnan(value) or np.isinf(value)
    else:
        value = float(value)
        return (value == float('inf')) or (value == float('-inf')) or (value == float('NaN'))


def replace_negative_inf(value):
    value = value.clone()
    value[value.data == -np.inf] = _log_epsilon
    return value


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


def truncate_str(s, length=50):
    return (s[:length] + '...') if len(s) > length else s


def is_hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def clamp_prob(prob):
    return torch.clamp(prob, min=_epsilon, max=1)


def clamp_log_prob(log_prob):
    return torch.clamp(log_prob, min=_log_epsilon)


def safe_torch_sum(t, *args, **kwargs):
    try:
        return torch.sum(t, *args, **kwargs)
    except RuntimeError:
        print(colored('Warning: torch.sum error (RuntimeError: value cannot be converted to type double without overflow) encountered, using tensor sum. Any gradient information through this variable will be lost.', 'red', attrs=['bold']))
        print(t)
        if isinstance(t, Variable):
            return Variable(Tensor([t.data.sum(*args, **kwargs)]))
        else:
            raise TypeError('Expecting a Variable.')


def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)


def weights_to_image(w):
    if not isinstance(w, np.ndarray):
        w = w.data.cpu().numpy()
    if w.ndim == 1:
        w = np.expand_dims(w, 1)
    if w.ndim > 2:
        c = w.shape[0]
        w = np.reshape(w, (c,-1))
    w_min = w.min()
    w_max = w.max()
    w -= w_min
    w *= (1/(w_max - w_min + _epsilon))
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(w)
    rgb_img = np.delete(rgba_img, 3,2)
    rgb_img = np.transpose(rgb_img,(2,0,1))
    return rgb_img


def rgb_blend(rgb1, rgb2, blend):
    # rgb1 and rgb2 are triples of (r, g, b) where r, g, b are between 0 and 1. blend is between 0 and 1.
    return rgb1[0] + (rgb2[0]-rgb1[0])*blend, rgb1[1] + (rgb2[1]-rgb1[1])*blend, rgb1[2] + (rgb2[2]-rgb1[2])*blend


def rgb_to_hex(rgb):
    # rgb is a triple of (r, g, b) where r, g, b are between 0 and 1.
    return "#{:02x}{:02x}{:02x}".format(int(max(0, min(rgb[0], 1))*255), int(max(0, min(rgb[1], 1))*255), int(max(0, min(rgb[2], 1))*255))


def crop_image(image_np):
    image_data_bw = image_np.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    return image_np[cropBox[0]:cropBox[1], cropBox[2]:cropBox[3], :]
