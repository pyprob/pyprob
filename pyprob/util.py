import torch
import numpy as np
import random
from termcolor import colored
import inspect
import sys

_device = torch.device('cpu')
_dtype = torch.float


# def is_hashable(v):
#     """Determine whether `v` can be hashed."""
#     try:
#         hash(v)
#     except TypeError:
#         return False
#     return True

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
