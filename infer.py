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

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--latest', help='show the latest artifact', action='store_true')
parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int)
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=4, type=int)
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

if not opt.latest and opt.nth is None:
    parser.print_help()
    quit()
