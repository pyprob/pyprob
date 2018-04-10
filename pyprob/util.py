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
    DEFAULT = 1  # Record traces, training data generation for inference network
    IMPORTANCE_SAMPLING_WITH_PRIOR = 2  # Record traces, importance sampling with proposals from prior
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 3  # Record traces, importance sampling with proposals from inference network
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 4  # Record traces for single-site Metropolis Hastings sampling, http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf and https://arxiv.org/abs/1507.00996


class InferenceEngine(enum.Enum):
    IMPORTANCE_SAMPLING = 0  # Type: IS
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 1  # Type: IS
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 2  # Type: MCMC


class Optimizer(enum.Enum):
    ADAM = 0
    SGD = 1


class InferenceNetworkTrainingMode(enum.Enum):
    USE_OBSERVE_DIST_SAMPLE = 0
    USE_OBSERVE_DIST_MEAN = 1


def set_random_seed(seed=123):
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
_cuda_device = -1


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


inference_network_training_mode = InferenceNetworkTrainingMode.USE_OBSERVE_DIST_SAMPLE


def set_inference_network_training_mode(mode=InferenceNetworkTrainingMode.USE_OBSERVE_DIST_SAMPLE):
    global inference_network_training_mode
    inference_network_training_mode = mode


def to_variable(value, requires_grad=False):
    ret = None
    if isinstance(value, Variable):
        ret = value
    elif torch.is_tensor(value):
        ret = Variable(value.float(), requires_grad=requires_grad)
    elif isinstance(value, np.ndarray):
        ret = Variable(torch.from_numpy(value.astype(float)).float(), requires_grad=requires_grad)
    elif value is None:
        ret = None
    elif isinstance(value, (list, tuple)):
        if isinstance(value[0], Variable):
            ret = torch.stack(value).float()
        elif torch.is_tensor(value[0]):
            ret = torch.stack(list(map(lambda x: Variable(x, requires_grad=requires_grad), value))).float()
        elif (type(value[0]) is float) or (type(value[0]) is int):
            ret = torch.stack(list(map(lambda x: Variable(Tensor([x]), requires_grad=requires_grad), value))).float().view(-1)
        else:
            ret = Variable(Tensor(value)).float()
    else:
        ret = Variable(Tensor([float(value)]), requires_grad=requires_grad)
    if _cuda_enabled:
        return ret.cuda()
    else:
        return ret


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


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('_%Y%m%d_%H%M%S')


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


def truncate_str(s, length=40):
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
        print('Warning: torch.sum error (RuntimeError: value cannot be converted to type double without overflow) encountered, using tensor sum. Any gradient information through this variable will be lost.')
        if isinstance(t, Variable):
            return Variable(Tensor([t.data.sum(*args, **kwargs)]))
        else:
            raise TypeError('Expecting a Variable.')


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

def _broadcast_shape(shapes):
    r"""
    Given a list of tensor sizes, returns the size of the resulting broadcasted
    tensor.
    Args:
        shapes (list of torch.Size): list of tensor sizes
    """
    shape = torch.Size()
    for s in shapes:
        shape = torch._C._infer_size(s, shape)
    return shape

def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.Tensor` and `torch.autograd.Variable` instances are broadcasted as
        per the `broadcasting rules
        <http://pytorch.org/docs/master/notes/broadcasting.html>`_
      - numbers.Number instances (scalars) are upcast to Variables having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to Variables having size
        `(1,)`.
    Args:
        values (list of `numbers.Number` or `torch.Tensor`)
    Raises:
        ValueError: if any of the values is not a `numbers.Number`, `torch.Tensor`
            or `torch.autograd.Variable` instance
    """
    values = list(values)
    scalar_idxs = [i for i in range(len(values)) if isinstance(values[i], Number)]
    tensor_idxs = [i for i in range(len(values)) if isinstance(values[i], torch.Tensor)]
    if len(scalar_idxs) + len(tensor_idxs) != len(values):
        raise ValueError('Input arguments must all be instances of numbers.Number or torch.Tensor.')
    if tensor_idxs:
        broadcast_shape = _broadcast_shape([values[i].size() for i in tensor_idxs])
        for idx in tensor_idxs:
            values[idx] = values[idx].expand(broadcast_shape)
        template = values[tensor_idxs[0]]
        if len(scalar_idxs) > 0 and not isinstance(template, torch.autograd.Variable):
            raise ValueError(('Input arguments containing instances of numbers.Number and torch.Tensor '
                              'are not currently supported.  Use torch.autograd.Variable instead of torch.Tensor'))
        for idx in scalar_idxs:
            values[idx] = template.new(template.size()).fill_(values[idx])
    else:
        for idx in scalar_idxs:
            values[idx] = torch.tensor(float(values[idx]))
    return values

class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value

def _sum_rightmost(value, dim):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.
    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    return value.contiguous().view(value.shape[:-dim] + (-1,)).sum(-1)
