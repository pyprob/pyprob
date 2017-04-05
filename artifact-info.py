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
from pprint import pformat
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Artifact Info)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--latest', help='show the latest artifact', action='store_true')
parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int)
parser.add_argument('--structure', help='show extra information about artifact structure', action='store_true')
parser.add_argument('--plotLoss', help='save loss plot to file (supported formats: eps, jpg, png, pdf, svg, tif)', type=str)
parser.add_argument('--plotLossToScreen', help='show the loss plot in screen', action='store_true')
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
util.log_print('Artifact Info')
util.log_print()
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print()
util.log_print('Running on PyTorch ' + torch.__version__)
util.log_print()
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))

util.log_print()
util.log_print(colored('█ Artifact info configuration', 'blue', attrs=['bold']))
util.log_print()
util.log_print(pformat(vars(opt)))
util.log_print()

if opt.latest:
    opt.nth = -1
file_name = util.file_starting_with('{0}/{1}'.format(opt.folder, 'compile-artifact'), opt.nth)
artifact = torch.load(file_name)
file_size = '{:,}'.format(os.path.getsize(file_name))

util.log_print()
util.log_print(colored('█ Artifact', 'blue', attrs=['bold']))
util.log_print()

util.check_versions(artifact)

util.log_print('File name             : {0}'.format(file_name))
util.log_print('File size (Bytes)     : {0}'.format(file_size))
util.log_print(artifact.get_info())

if opt.structure:
    util.log_print()
    util.log_print(colored('█ Artifact structure', 'blue', attrs=['bold']))
    util.log_print()

    util.log_print(artifact.get_structure())

if opt.plotLossToScreen or opt.plotLoss:
    util.log_print()
    util.log_print(colored('█ Loss plot', 'blue', attrs=['bold']))
    util.log_print()
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(artifact.valid_history_trace, artifact.valid_history_loss, label='Validation')
    ax.plot(artifact.train_history_trace, artifact.train_history_loss, label='Training')
    ax.legend()
    plt.xlabel('Traces')
    plt.ylabel('Loss')
    plt.grid()
    fig.tight_layout()
    if opt.plotLossToScreen:
        util.log_print('Plotting loss to screen')
        plt.show()
    if not opt.plotLoss is None:
        util.log_print('Saving loss plot to file: ' + opt.plotLoss)
        fig.savefig(opt.plotLoss)
