import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import datetime

_random_seed = 0
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
def set_cuda(cuda, device=0):
    global Tensor
    global _cuda_enabled
    global _cuda_device
    if torch.cuda.is_available() and cuda:
        _cuda_enabled = True
        _cuda_device = device
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        Tensor = torch.cuda.FloatTensor
    else:
        _cuda_enabled = False
        Tensor = torch.FloatTensor

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
            ret = torch.stack(list(map(lambda x:Variable(x, requires_grad=requires_grad), value))).float()
        elif (type(value[0]) is float) or (type(value[0]) is int):
            ret = torch.stack(list(map(lambda x:Variable(Tensor([x]), requires_grad=requires_grad), value))).float().view(-1)
        else:
            ret = Variable(torch.Tensor(value)).float()
    else:
        ret = Variable(Tensor([float(value)]), requires_grad=requires_grad)
    if _cuda_enabled:
        return ret.cuda()
    else:
        return ret

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

def debug(*expressions):
    frame = sys._getframe(1)
    print()
    max_str_length = 0
    for expression in expressions:
        if len(expression) > max_str_length:
            max_str_length = len(expression)
    for expression in expressions:
        print('  {} = {}'.format(expression.ljust(max_str_length), repr(eval(expression, frame.f_globals, frame.f_locals))))

def pack_observes_to_variable(observes):
    try:
        return torch.stack([to_variable(o) for o in observes])
    except:
        try:
            return torch.cat([to_variable(o).view(-1) for o in observes])
        except:
            return to_variable(Tensor())

def one_hot(dim, i):
    t = Tensor(dim).zero_()
    t.narrow(0, i, 1).fill_(1)
    return to_variable(t)

def kl_divergence_normal(p_mean, p_stddev, q_mean, q_stddev):
    p_mean = to_variable(p_mean)
    p_stddev = to_variable(p_stddev)
    q_mean = to_variable(q_mean)
    q_stddev = to_variable(q_stddev)
    return torch.log(q_stddev) - torch.log(p_stddev) + (p_stddev.pow(2) + (p_mean - q_mean).pow(2)) / (2 * q_stddev.pow(2)) - 0.5

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
