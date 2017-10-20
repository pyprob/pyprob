#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import time
import datetime
import logging
import sys
from termcolor import colored
from threading import Thread
import cpuinfo
from glob import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg') # Do not use X server
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import pyprob
from pyprob.logger import Logger

random_seed = 0
def set_random_seed(seed):
    global random_seed
    random_seed = seed
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
set_random_seed(123)

Tensor = torch.FloatTensor
cuda_enabled = False
cuda_device = -1
epsilon = 1e-8
beta_res = 1000
beta_step = 1/beta_res
beta_integration_domain = Variable(torch.linspace(beta_step/2,1-(beta_step/2),beta_res), requires_grad=False)

def in_jupyter():
    try:
        get_ipython
        return True
    except:
        return False

logger = Logger()

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')

def get_config():
    ret = []
    ret.append(colored('pyprob  {}'.format(pyprob.__version__), 'blue', attrs=['bold']))
    ret.append('PyTorch {}'.format(torch.__version__))
    cpu_info = cpuinfo.get_cpu_info()
    if 'brand' in cpu_info:
        ret.append('CPU           : {}'.format(cpu_info['brand']))
    else:
        ret.append('CPU           : unknown')
    if 'count' in cpu_info:
        ret.append('CPU count     : {0} (logical)'.format(cpu_info['count']))
    else:
        ret.append('CPU count     : unknown')
    if torch.cuda.is_available():
        ret.append('CUDA          : available')
        ret.append('CUDA devices  : {0}'.format(torch.cuda.device_count()))
        if cuda_enabled:
            if cuda_device == -1:
                ret.append('CUDA selected : all')
            else:
                ret.append('CUDA selected : {0}'.format(cuda_device))
    else:
        ret.append('CUDA          : not available')
    if cuda_enabled:
        ret.append('Running on    : CUDA')
    else:
        ret.append('Running on    : CPU')
    return '\n'.join(ret)

def set_cuda(cuda, device=0):
    global cuda_enabled
    global cuda_device
    global Tensor
    global beta_integration_domain
    if torch.cuda.is_available() and cuda:
        cuda_enabled = True
        cuda_device = device
        torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        Tensor = torch.cuda.FloatTensor
        beta_integration_domain = beta_integration_domain.cuda()
    else:
        cuda_enabled = False
        Tensor = torch.FloatTensor
        beta_integration_domain = beta_integration_domain.cpu()

def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_artifact(file_name, cuda=False, device_id=-1):
    try:
        if cuda:
            artifact = torch.load(file_name)
        else:
            artifact = torch.load(file_name, map_location=lambda storage, loc: storage)
    except:
        logger.log_error('load_artifact: Cannot load file')
    if artifact.code_version != pyprob.__version__:
        logger.log()
        logger.log_warning('Different pyprob versions (artifact: {0}, current: {1})'.format(artifact.code_version, pyprob.__version__))
        logger.log()
    if artifact.pytorch_version != torch.__version__:
        logger.log()
        logger.log_warning('Different PyTorch versions (artifact: {0}, current: {1})'.format(artifact.pytorch_version, torch.__version__))
        logger.log()

    # if print_info:
    #     file_size = '{:,}'.format(os.path.getsize(file_name))
    #     log_print('File name             : {0}'.format(file_name))
    #     log_print('File size (Bytes)     : {0}'.format(file_size))
    #     log_print(artifact.get_info())
    #     log_print()

    if cuda:
        if device_id == -1:
            device_id = torch.cuda.current_device()

        if artifact.on_cuda:
            if device_id != artifact.cuda_device_id:
                logger.log_warning('Loading CUDA (device {0}) artifact to CUDA (device {1})'.format(artifact.cuda_device_id, device_id))
                logger.log()
                artifact.move_to_cuda(device_id)
        else:
            logger.log_warning('Loading CPU artifact to CUDA (device {0})'.format(device_id))
            logger.log()
            artifact.move_to_cuda(device_id)
    else:
        if artifact.on_cuda:
            logger.log_warning('Loading CUDA artifact to CPU')
            logger.log()
            artifact.move_to_cpu()

    return artifact

def save_artifact(artifact, file_name):
    sys.stdout.write('Updating artifact on disk...                             \r')
    sys.stdout.flush()
    artifact.modified = get_time_str()
    artifact.updates += 1
    artifact.trained_on = 'CUDA' if cuda_enabled else 'CPU'
    def thread_save():
        torch.save(artifact, file_name)
    a = Thread(target=thread_save)
    a.start()
    a.join()

def file_starting_with(pattern, n):
    try:
        ret = sorted(glob(pattern + '*'))[n]
    except:
        logger.log_error('Cannot find file')
        sys.exit(1)
    return ret

def truncate_str(s, length=80):
    return (s[:length] + '...') if len(s) > length else s

def one_hot(dim, i):
    t = Tensor(dim).zero_()
    t.narrow(0, i, 1).fill_(1)
    return t

def to_tensor(value):
    if torch.is_tensor(value):
        return value
    elif type(value) is float:
        return Tensor([value])
    elif type(value) is int:
        return Tensor([value])
    else:
        util.logger.log_error('Unexpected type.')

def days_hours_mins_secs(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))

def pack_repetitions(l):
    if len(l) == 1:
        return [(l[0], 1)]
    else:
        ret = []
        prev = l[0]
        prev_count = 1
        for i in range(1,len(l)):
            if l[i] == prev:
                prev_count += 1
            else:
                ret.append((prev, prev_count))
                prev = l[i]
                prev_count = 1
        ret.append((prev, prev_count))
        return ret

def rgb_blend(rgb1,rgb2,blend):
    # rgb1 and rgb2 are triples of (r, g, b) where r, g, b are between 0 and 1. blend is between 0 and 1.
    return rgb1[0] + (rgb2[0]-rgb1[0])*blend, rgb1[1] + (rgb2[1]-rgb1[1])*blend, rgb1[2] + (rgb2[2]-rgb1[2])*blend

def rgb_to_hex(rgb):
    # rgb is a triple of (r, g, b) where r, g, b are between 0 and 1.
    return "#{:02x}{:02x}{:02x}".format(int(max(0,min(rgb[0],1))*255),int(max(0,min(rgb[1],1))*255),int(max(0,min(rgb[2],1))*255))

def crop_image(image_np):
    image_data_bw = image_np.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    return image_np[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

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
    w *= (1/(w_max - w_min + epsilon))
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(w)
    rgb_img = np.delete(rgba_img, 3,2)
    rgb_img = np.transpose(rgb_img,(2,0,1))
    return rgb_img

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

def beta(a, b):
    n = a.nelement()
    assert b.nelement() == n, 'a and b must have the same number of elements'
    a_min_one = (a-1).repeat(beta_res,1).t()
    b_min_one = (b-1).repeat(beta_res,1).t()
    x = beta_integration_domain.repeat(n,1)
    fx = (x**a_min_one) * ((1-x)**b_min_one)
    return torch.sum(fx,1).squeeze() * beta_step
