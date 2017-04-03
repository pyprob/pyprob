#
# Oxford Inference Compilation
# Compilation mode
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

from __future__ import print_function
import util
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
import logging
import sys
import datetime

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--out', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=1)
parser.add_argument('--learningRate', help='learning rate', default=0.0001)
parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005)
parser.add_argument('--batchSize', help='training batch size', default=128)
parser.add_argument('--validSize', help='validation set size', default=256)
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.out, 'compile-log' + time_stamp))

util.print_log(colored('Oxford Inference Compilation ' + util.version, 'white', 'on_blue', attrs=['bold']))
util.print_log('Compilation Mode')
util.print_log('')
util.print_log('Started ' +  str(datetime.datetime.now()))
util.print_log('')
util.print_log('Running on PyTorch')
util.print_log('')


torch.manual_seed(opt.seed)
if opt.cuda:
    if not torch.cuda.is_available():
        util.print_log(colored('Error: CUDA not available', 'red'))
        quit()
    torch.cuda.manual_seed(opt.seed)
