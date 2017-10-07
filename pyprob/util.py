import time
import datetime
import logging
import sys
from termcolor import colored
from threading import Thread
import cpuinfo
from glob import glob

import torch
from torch.autograd import Variable
import pyprob
from pyprob.logger import Logger

Tensor = torch.FloatTensor
random_seed = 123
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

logger = Logger('{0}/{1}'.format('.', 'pyprob-log' + get_time_stamp()))

def set_cuda(cuda, device=0):
    global cuda_enabled
    global cuda_device
    global Tensor
    global beta_integration_domain

    if torch.cuda.is_available() and cuda:
        cuda_enabled = True
        cuda_device = device
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(random_seed)
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

def days_hours_mins_secs(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))

def beta(a, b):
    n = a.nelement()
    assert b.nelement() == n, 'a and b must have the same number of elements'
    a_min_one = (a-1).repeat(beta_res,1).t()
    b_min_one = (b-1).repeat(beta_res,1).t()
    x = beta_integration_domain.repeat(n,1)
    fx = (x**a_min_one) * ((1-x)**b_min_one)
    return torch.sum(fx,1).squeeze() * beta_step
