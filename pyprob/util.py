import torch
import numpy as np
import random
from termcolor import colored
import inspect
import sys
import enum
import time
import math
from functools import reduce
import operator
import datetime

from .distributions import Categorical


_device = torch.device('cpu')
_dtype = torch.float
_cuda_enabled = False
_verbosity = 2
_print_refresh_rate = 0.25  # seconds
_epsilon = 1e-8
_log_epsilon = math.log(_epsilon)  # log(1e-8) = -18.420680743952367


class TraceMode(enum.Enum):
    PRIOR = 1
    POSTERIOR = 2


class PriorInflation(enum.Enum):
    DISABLED = 0
    ENABLED = 1


class InferenceEngine(enum.Enum):
    IMPORTANCE_SAMPLING = 0  # Type: IS; Importance sampling with proposals from prior
    IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK = 1  # Type: IS; Importance sampling with proposals from inference network
    LIGHTWEIGHT_METROPOLIS_HASTINGS = 2  # Type: MCMC; Lightweight (single-site) Metropolis Hastings sampling, http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf and https://arxiv.org/abs/1507.00996
    RANDOM_WALK_METROPOLIS_HASTINGS = 3  # Type: MCMC; Lightweight Metropolis Hastings with single-site proposal kernels that depend on the value of the site


class InferenceNetwork(enum.Enum):
    FEEDFORWARD = 0


class ObserveEmbedding(enum.Enum):
    FEEDFORWARD = 0
    CNN2D5C = 1
    CNN3D4C = 2


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


def set_cuda(enabled, device=None):
    global _device
    global _cuda_enabled
    if torch.cuda.is_available() and enabled:
        _device = torch.device('cuda')
        _cuda_enabled = True
    else:
        _device = torch.device('cpu')
        _cuda_enabled = False


def set_verbosity(v=2):
    global _verbosity
    _verbosity = v


def to_tensor(value, dtype=None):
    if dtype is None:
        dtype = _dtype
    return torch.tensor(value).to(device=_device, dtype=dtype)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except:
            raise TypeError('Cannot convert to Numpy array.')


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


def progress_bar(i, len):
    bar_len = 20
    filled_len = int(round(bar_len * i / len))
    # percents = round(100.0 * i / len, 1)
    return '#' * filled_len + '-' * (bar_len - filled_len)


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def has_nan_or_inf(value):
    if torch.is_tensor(value):
        value = torch.sum(value)
        isnan = int(torch.isnan(value)) > 0
        isinf = int(torch.isinf(value)) > 0
        return isnan or isinf
    else:
        value = float(value)
        return (value == float('inf')) or (value == float('-inf')) or (value == float('NaN'))


def replace_negative_inf(value):
    value = value.clone()
    value[value == -np.inf] = _log_epsilon
    return value


def rgb_to_hex(rgb):
    # rgb is a triple of (r, g, b) where r, g, b are between 0 and 1.
    return "#{:02x}{:02x}{:02x}".format(int(max(0, min(rgb[0], 1))*255), int(max(0, min(rgb[1], 1))*255), int(max(0, min(rgb[2], 1))*255))


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def truncate_str(s, length=50):
    return (s[:length] + '...') if len(s) > length else s


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')


def one_hot(dim, i):
    t = torch.zeros(dim)
    t.narrow(0, i, 1).fill_(1)
    return t


def is_hashable(v):
    try:
        hash(v)
    except TypeError:
        return False
    return True


<<<<<<< HEAD
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
=======
def empirical_to_categorical(empirical_dist, max_val=None):
    empirical_dist = empirical_dist.combine_duplicates().map(int)
    if max_val is None:
        max_val = int(empirical_dist.max)
    probs = torch.zeros(max_val + 1)
    for i in range(empirical_dist.length):
        val = empirical_dist.values[i]
        if val <= max_val:
            probs[val] = float(empirical_dist.weights[i])
    return Categorical(probs)
>>>>>>> origin/master
