#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import infcomp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import logging
import sys
import re
from glob import glob
from termcolor import colored

epsilon = 1e-8

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')

def init(opt):
    global Tensor
    torch.manual_seed(opt.seed)
    if opt.cuda:
        if not torch.cuda.is_available():
            log_error('CUDA not available')
        torch.cuda.manual_seed(opt.seed)
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

def init_logger(file_name):
    global logger
    logger = logging.getLogger()
    logger_file_handler = logging.FileHandler(file_name)
    logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logger_file_handler)
    logger.setLevel(logging.INFO)

def log_print(line=''):
    print(line)
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    logger.info(ansi_escape.sub('', line))

def log_error(line):
    print(colored('Error: ' + line, 'red', attrs=['bold']))
    logger.error('Error: ' + line)
    quit()

def log_warning(line):
    print(colored('Warning: ' + line, 'red', attrs=['bold']))
    logger.warning('Warning: ' + line)

def standardize(t):
    mean = torch.mean(t)
    sd = torch.std(t)
    t.add_(-mean)
    t.div_(sd + epsilon)
    return t

def days_hours_mins_secs(delta):
    s = delta.total_seconds()
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(delta.days, int(h), int(m), int(s))

def file_starting_with(pattern, n):
    return sorted(glob(pattern + '*'))[n]

def check_versions(artifact):
    if artifact.code_version != infcomp.__version__:
        log_print()
        log_warning('Different code versions (artifact: {0}, current: {1})'.format(artifact.code_version, infcomp.__version__))
        log_print()
    if artifact.pytorch_version != torch.__version__:
        log_print()
        log_warning('Different PyTorch versions (artifact: {0}, current: {1})'.format(artifact.pytorch_version, torch.__version__))
        log_print()

class Spinner(object):
    def __init__(self):
        self.i = 0
        self.spinner = [colored('█   \r', 'blue', attrs=['bold']),
                        colored(' █  \r', 'blue', attrs=['bold']),
                        colored('  █ \r', 'blue', attrs=['bold']),
                        colored('   █\r', 'blue', attrs=['bold']),
                        colored('  █ \r', 'blue', attrs=['bold']),
                        colored(' █  \r', 'blue', attrs=['bold'])]
    def spin(self):
        sys.stdout.write(self.spinner[self.i])
        sys.stdout.flush()
        self.i +=1
        if self.i > 5:
            self.i = 0
