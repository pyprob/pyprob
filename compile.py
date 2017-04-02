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

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('-o', '--out', help='folder to save artifacts and logs', default='./artifacts')
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
