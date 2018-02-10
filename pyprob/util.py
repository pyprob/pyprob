import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

Tensor = torch.FloatTensor

def to_variable(value, requires_grad=False):
    if isinstance(value, Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value.float(), requires_grad=requires_grad)
    elif isinstance(value, np.ndarray):
        return Variable(torch.from_numpy(value.astype(float)).float(), requires_grad=requires_grad)
    elif value is None:
        return
    elif isinstance(value, (list, tuple)):
        if isinstance(value[0], Variable):
            return torch.stack(value)
        elif torch.is_tensor(value[0]):
            return torch.stack(list(map(Variable, value)))
        else:
            return torch.stack(list(map(lambda x:Variable(Tensor(x)), value)))
    else:
        return Variable(torch.Tensor([float(value)]), requires_grad=requires_grad)

def logsumexp(x, dim=0):
    '''
    https://en.wikipedia.org/wiki/LogSumExp
    input:
        x: Tensor/Variable [dim_1 * dim_2 * ... * dim_N]
        dim: n
    output: Tensor/Variable [dim_1 * ... * dim_{n - 1} * 1 * dim_{n + 1} * ... * dim_N]
    '''
    x_max, _ = x.max(dim)
    x_diff = x - x_max.expand_as(x)
    return x_max + x_diff.exp().sum(dim).log()

def fast_np_random_choice(values, probs):
    # See https://mobile.twitter.com/RadimRehurek/status/928671225861296128
    probs /= probs.sum()
    return values[np.searchsorted(probs.cumsum(), random.random())]

def debug(expression1, expression2):
    frame = sys._getframe(1)
    print('\n  {} = {}; {} = {}'.format(expression1, repr(eval(expression1, frame.f_globals, frame.f_locals)), expression2, repr(eval(expression2, frame.f_globals, frame.f_locals))))
