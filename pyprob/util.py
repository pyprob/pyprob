import torch
import numpy as np
import random
from termcolor import colored
import inspect
import sys
import enum

_device = torch.device('cpu')
_dtype = torch.float
_verbosity = 2
_print_refresh_rate = 0.2  # seconds


class TraceMode(enum.Enum):
    NONE = 0  # No trace recording, forward sample trace results only
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


def set_verbosity(v=2):
    global _verbosity
    _verbosity = v


def to_tensor(value):
    return torch.tensor(value).to(device=_device, dtype=_dtype)


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
