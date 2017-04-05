#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import util
from protocol import Replier

import torch
import argparse
from termcolor import colored
import datetime
import sys
from pprint import pformat

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--latest', help='show the latest artifact', action='store_true')
parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int)
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=4, type=int)
parser.add_argument('--debug', help='show debugging information as requests arrive', action='store_true')
parser.add_argument('--server', help='address and port to bind this inference serve', default='tcp://*:6666')
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

if not opt.latest and opt.nth is None:
    parser.print_help()
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.folder, 'artifact-info-log' + time_stamp))

util.log_print()
util.log_print(colored('█ Oxford Inference Compilation ' + util.version, 'blue', attrs=['bold']))
util.log_print()
util.log_print('Inference Mode')
util.log_print()
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print()
util.log_print('Running on PyTorch ' + torch.__version__)
util.log_print()
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))

util.log_print()
util.log_print(colored('█ Inference configuration', 'blue', attrs=['bold']))
util.log_print()
util.log_print(pformat(vars(opt)))
util.log_print()

with Replier(opt.server) as replier:

    print('test')

print('test2')
