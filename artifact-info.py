#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import util

import torch
import argparse
from termcolor import colored
import datetime
import sys
import os

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Artifact Info)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.folder, 'artifact-info-log' + time_stamp))

util.log_print()
util.log_print(colored('█ Oxford Inference Compilation ' + util.version, 'blue', attrs=['bold']))
util.log_print()
util.log_print('Artifact Info')
util.log_print()
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print()
util.log_print('Running on PyTorch ' + torch.__version__)
util.log_print()
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))

file_name = util.file_starting_with('{0}/{1}'.format(opt.folder, 'compile-artifact'), -1)
artifact = torch.load(file_name)
file_size = '{:,}'.format(os.path.getsize(file_name))


util.log_print()
util.log_print(colored('█ Artifact', 'blue', attrs=['bold']))
util.log_print()

util.log_print('File name             : {0}'.format(file_name))
util.log_print('File size (Bytes)     : {0}'.format(file_size))
util.log_print(artifact.get_info())
