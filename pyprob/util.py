import torch
from torch import nn
import numpy as np
import random
from termcolor import colored
import inspect
import os
import sys
import enum
import time
import math
from functools import reduce
import operator
import datetime
import inspect
import torch.multiprocessing
from scipy.stats import wasserstein_distance as wd

from .distributions import Categorical

torch.multiprocessing.set_sharing_strategy('file_system')

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
    PRIOR_FOR_INFERENCE_NETWORK = 3


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
    LSTM = 1


class ObserveEmbedding(enum.Enum):
    FEEDFORWARD = 0
    CNN2D5C = 1
    CNN3D5C = 2


class Optimizer(enum.Enum):
    ADAM = 0
    SGD = 1
    ADAM_LARC = 2
    SGD_LARC = 3


class LearningRateScheduler(enum.Enum):
    NONE = 0
    POLY1 = 1
    POLY2 = 2


class ImportanceWeighting(enum.Enum):
    IW0 = 0  # use prior as proposal for all accepted=False addresses
    IW1 = 1  # Discard all the rejected samples
    IW2 = 2  # Do not discard anything


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_rng_state():
    # Get combined state for all packages set in set_random_seed.
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state()}

def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])

def set_device(device='cpu'):
    global _device
    global _cuda_enabled
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            _device = device
            _cuda_enabled = True
        else:
            print(colored('Warning: cannot enable CUDA device: {}'.format(device), 'red', attrs=['bold']))
    else:
        _device = device
        _cuda_enabled = False
    try:
        test = to_tensor(1.)
        test.to(device=_device)
    except Exception as e:
        print(e)
        raise RuntimeError('Cannot set device: {}'.format(device))


def set_verbosity(v=2):
    global _verbosity
    _verbosity = v


def to_tensor(value, dtype=_dtype):
    if not torch.is_tensor(value):
        if type(value) == np.int64:
            value = torch.tensor(float(value))
        elif type(value) == np.float32:
            value = torch.tensor(float(value))
        elif type(value) == list:
            value = torch.tensor(value)
        else:
            value = torch.tensor(value)

    return value.to(dtype=dtype)

def to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        try:
            return np.array(value)
        except Exception as e:
            print(e)
            raise TypeError('Cannot convert to Numpy array.')


def to_size(value):
    if isinstance(value, torch.Size):
        return value
    elif isinstance(value, int):
        return torch.Size([value])
    elif isinstance(value, list):
        return torch.Size(value)
    else:
        raise TypeError('Expecting a torch.Size, int, or list of ints.')


def fast_np_random_choice(values, probs_cumsum):
    # See https://mobile.twitter.com/RadimRehurek/status/928671225861296128
    return values[min(np.searchsorted(probs_cumsum, random.random()), len(values) - 1)]


def eval_print(*expressions):
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


progress_bar_num_iters = None
progress_bar_len_str_num_iters = None
progress_bar_time_start = None
progress_bar_prev_duration = None


def progress_bar_init(message, num_iters, iter_name='Items'):
    global progress_bar_num_iters
    global progress_bar_len_str_num_iters
    global progress_bar_time_start
    global progress_bar_prev_duration
    if num_iters < 1:
        raise ValueError('num_iters must be a positive integer')
    progress_bar_num_iters = num_iters
    progress_bar_time_start = time.time()
    progress_bar_prev_duration = 0
    progress_bar_len_str_num_iters = len(str(progress_bar_num_iters))
    print(message)
    sys.stdout.flush()
    print('Time spent  | Time remain.| Progress             | {} | {}/sec'.format(iter_name.ljust(progress_bar_len_str_num_iters * 2 + 1), iter_name))


def progress_bar_update(iter):
    global progress_bar_prev_duration
    duration = time.time() - progress_bar_time_start
    if (duration - progress_bar_prev_duration > _print_refresh_rate) or (iter >= progress_bar_num_iters - 1):
        progress_bar_prev_duration = duration
        traces_per_second = (iter + 1) / duration
        print('{} | {} | {} | {}/{} | {:,.2f}       '.format(days_hours_mins_secs_str(duration), days_hours_mins_secs_str((progress_bar_num_iters - iter) / traces_per_second), progress_bar(iter, progress_bar_num_iters), str(iter).rjust(progress_bar_len_str_num_iters), progress_bar_num_iters, traces_per_second), end='\r')
        sys.stdout.flush()


def progress_bar_end(message=None):
    progress_bar_update(progress_bar_num_iters)
    print()
    if message is not None:
        print(message)


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


def safe_log(value):
    value = torch.log(value)
    if torch.any(value == -np.inf):
        return replace_negative_inf(value)
    else:
        return value


def replace_inf(value, replace_message=None):
    if torch.any(value == np.inf):
        value = value.clone()
        value[value == np.inf] = 0.
        if replace_message is not None:
            print(replace_message)
    return value


def replace_negative_inf(value, replace_message=None):
    if torch.any(value == -np.inf):
        value = value.clone()
        value[value == -np.inf] = _log_epsilon
        if replace_message is not None:
            print(replace_message)
    return value


def rgb_to_hex(rgb):
    # rgb is a triple of (r, g, b) where r, g, b are between 0 and 1.
    return "#{:02x}{:02x}{:02x}".format(int(max(0, min(rgb[0], 1))*255), int(max(0, min(rgb[1], 1))*255), int(max(0, min(rgb[2], 1))*255))


def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))


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
    return to_tensor(t)


def is_hashable(v):
    try:
        hash(v)
    except TypeError:
        return False
    return True


def empirical_to_categorical(empirical_dist, max_val=None):
    empirical_dist = empirical_dist.combine_duplicates().map(int)
    if max_val is None:
        max_val = int(empirical_dist.max)
    probs = torch.zeros(max_val + 1)
    for i in range(empirical_dist.length):
        val = empirical_dist._get_value(i)
        if val <= max_val:
            probs[val] = float(empirical_dist._get_weight(i))
    return Categorical(probs)


def check_gnu_dbm():
    try:
        import dbm.gnu
    except ModuleNotFoundError:
        return False
    return True


def tile_rows_cols(num_items):
    cols = math.ceil(math.sqrt(num_items))
    rows = 0
    while num_items > 0:
        rows += 1
        num_items -= cols
    return rows, cols


def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        os.makedirs(dir)


def address_id_to_int(address_id):
    if '__' not in address_id:
        return 0.
    else:
        divider_i = address_id.find('__')
        value_id = address_id[1:divider_i]
        return int(value_id)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# From https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py
def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)

def clamp_logits(logits):
    eps = torch.finfo(logits.dtype).eps
    return logits.clamp(min=torch.log(torch.tensor(eps)).item(),
                        max=torch.log1p(torch.Tensor([-eps])).item())


def init_distributed_print(rank, world_size, debug_print=True):
    if not debug_print:
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        # labelled print with info of [rank/world_size]
        old_out = sys.stdout

        class LabeledStdout:
            def __init__(self, rank, world_size):
                self._rank = rank
                self._world_size = world_size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[r%d/ws%d] %s' % (self._rank, self._world_size, x))

        sys.stdout = LabeledStdout(rank, world_size)


def drop_items(l, num_items_to_drop):
    if num_items_to_drop > len(l):
        raise ValueError('Cannot drop more items than the list length')
    ret = l.copy()
    for _ in range(num_items_to_drop):
        del(ret[random.randrange(len(ret))])
    return ret


def get_source(obj):
    try:
        return inspect.getsource(obj)
    except:
        return obj.__name__

def to_variable_dict_data(variable, variables_observed_inf_training=[],
                          to_list=True):
    """ sub_trace_data dictionary

    Inputs:

    trace_attr_dict -- dictionary in which to store attributes
    attr            -- str - attribute
    to_list         -- boolean specifying converting to a list for json serialization

    We further decode the json string into bytes (which has to be decoded once we load the data again)

    """
    VARIABLE_ATTRIBUTES = ['value', 'address_base', 'address', 'instance', 'log_prob',
                           'log_importance_weight', 'control', 'name', 'observed',
                           'tagged', 'constants', 'accepted', 'reused', 'distribution_name',
                           'distribution_args']
    var_dict = {}
    for attr in VARIABLE_ATTRIBUTES:
        if attr == 'value':
            v = getattr(variable, attr)
            var_dict[attr] = v.tolist() if to_list else v
        elif attr in ['distribution_name']:
            # extract the input arguments for initializing the distribution
            var_dict[attr] = variable.distribution_name
        elif attr in ['distribution_args']:
            tmp = {}
            for k, v in variable.distribution_args.items():
                tmp[k] = v.tolist() if to_list else v
            var_dict[attr] = tmp
        elif attr in ['constants']:
            tmp = {}
            for k, value in variable.constants.items():
                tmp[k] = value.tolist() if to_list else value
            var_dict[attr] = tmp
        elif attr in ['log_prob']:
            v = getattr(variable, attr)
            var_dict[attr] = v.item() if to_list else v
        elif attr in ['observed']:
            var_dict[attr] = getattr(variable, attr) or (getattr(variable, 'name') in variables_observed_inf_training)
        else:
            var_dict[attr] = getattr(variable, attr)

    return var_dict

def from_variable_dict_data(list_of_variable_dict, variables_observed_inf_training=[]):

        trace_list = []

        for variable_attr_dict in list_of_variable_dict:
            var_args = {}
            for attr, variable_data in variable_attr_dict.items():
                if attr == 'value':
                    var_args[attr] = to_tensor(variable_data)
                elif attr in ['distribution_name']:
                    # extract the input arguments for initializing the distribution
                    var_args[attr] = variable_data
                elif attr in ['distribution_args']:
                    tmp = {}
                    for k, value in variable_data.items():
                        tmp[k] = to_tensor(value)
                    var_args[attr] = tmp
                elif attr in ['constants']:
                    tmp = {}
                    for k, value in variable_data.items():
                        tmp[k] = to_tensor(value)
                    var_args[attr] = tmp
                elif attr in ['log_prob']:
                    var_args[attr] = to_tensor(variable_data)
                elif attr in ['observed']:
                    var_args[attr] = variable_data or variable_attr_dict['name'] in variables_observed_inf_training
                else:
                    var_args[attr] = variable_data

            trace_list.append(var_args)
        return trace_list

class RunningMoments(nn.Module):
    def __init__(self, initial_x=None):
        super().__init__()
        self.seen = 0
        self.mean =  nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        if initial_x is not None:
            self.update(initial_x)

    def update(self, x):
        w = len(x) / (self.seen+len(x))
        self.mean = nn.Parameter(self.mean*(1-w) + x.mean(dim=0)*w, requires_grad=False)
        self.std = nn.Parameter(self.std*(1-w) + w*((x-self.mean)**2).mean(dim=0)**0.5, requires_grad=False)
        self.seen = self.seen + len(x)

    def get(self):
        return self.mean, self.std+1e-8

def constants_to_tensors(constants):
    tmp = {}
    if constants:
        for k, v in constants.items():
            try:
                attr_shape = getattr(distribution, k).shape
                tmp[k] = util.to_tensor(v).view(attr_shape)
            except Exception as e:
                raise ValueError("Values in constant for distribution {} cannot be made a tensor".format(distribution.name))
    return tmp

def earthmover_distance_categorical(p1, p2):
    if not isinstance(p1, Categorical) or not isinstance(p2, Categorical):
        raise ValueError('Input distributions must be Categoricals')

    probs1 = p1._probs
    probs2 = p2._probs

    return wd(range(len(probs1)), range(len(probs2)), probs1, probs2)