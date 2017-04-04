#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import util
from modules import Sample, Trace, Artifact
from protocol import Requester

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
parser.add_argument('--server', help='address of the probprog model server', default='tcp://127.0.0.1:5555')
parser.add_argument('--learningRate', help='learning rate', default=0.0001, type=float)
parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005, type=float)
parser.add_argument('--batchSize', help='training batch size', default=128, type=int)
parser.add_argument('--validSize', help='validation set size', default=256, type=int)
parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64, type=int)
parser.add_argument('--noStandardize', help='do not standardize observations', action='store_true')
parser.add_argument('--resume', help='resume training of the latest artifact', action='store_true')
parser.add_argument('--obsEmb', help='observation embedding', choices=['fc'], default='fc', type=str)
parser.add_argument('--obsEmbDim', help='observation embedding dimension', default=512, type=int)
parser.add_argument('--smpEmb', help='sample embedding', choices=['fc'], default='fc', type=str)
parser.add_argument('--smpEmbDim', help='sample embedding dimension', default=1, type=int)
parser.add_argument('--lstmDim', help='lstm hidden unit dimension', default=512, type=int)
parser.add_argument('--lstmDepth', help='number of stacked lstms', default=1, type=int)
parser.add_argument('--softmaxBoost', help='multiplier before softmax', default=20.0, type=float)
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.out, 'compile-log' + time_stamp))
util.init(opt)

util.log_print(colored('Oxford Inference Compilation ' + util.version, 'white', 'on_blue', attrs=['bold']))
util.log_print('Compilation Mode')
util.log_print('')
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print('')
util.log_print('Running on PyTorch')
util.log_print('')
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))
util.log_print('')

with Requester(opt.server) as requester:

    artifact = Artifact()
    if not opt.resume:
        artifact.standardize = not opt.noStandardize
        artifact.cuda = opt.cuda
        artifact.one_hot_address_dim = opt.oneHotDim
        artifact.one_hot_instance_dim = opt.oneHotDim
        artifact.one_hot_proposal_type_dim = 1
        artifact.valid_size = opt.validSize
        requester.request_batch(artifact.valid_size)
        artifact.valid_batch = requester.receive_batch(artifact.standardize)
        artifact.lstm_dim = opt.lstmDim
        artifact.lstm_depth = opt.lstmDepth
        artifact.smp_emb = opt.smpEmb
        artifact.smp_emb_dim = opt.smpEmbDim
        artifact.obs_emb = opt.obsEmb
        artifact.obs_emb_dim = opt.obsEmbDim
        artifact.softmax_boost = opt.softmaxBoost
        artifact.polymorph()

    # torch.save(artifact, 'art1')
    # artifact = torch.load('art1')
    # print(artifact)

    print(artifact.valid_batch)
    print(artifact.sample_layers)
    print(artifact.proposal_layers)
