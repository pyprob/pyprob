#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import logging
import sys
import re

from termcolor import colored

version = '0.9.1'
epsilon = 1e-5

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')

def init(opt):
    global Tensor
    torch.manual_seed(opt.seed)
    if opt.cuda:
        if not torch.cuda.is_available():
            util.log_print(colored('Error: CUDA not available', 'red'))
            quit()
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

def log_print(line):
    print(line)
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    logger.info(ansi_escape.sub('', line))

def log_error(line):
    print(colored('Error: ' + line, 'red'))
    logger.error('Error: ' + line)
    quit()

def log_warning(line):
    print(colored('Warning: ' + line, 'yellow'))
    logger.warning('Warning: ' + line)

def standardize(t):
    mean = torch.mean(t)
    sd = torch.std(t)
    t.add_(-mean)
    t.div_(sd + epsilon)
    return t
