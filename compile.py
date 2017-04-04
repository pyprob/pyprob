#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

from __future__ import print_function
import util
from util import Sample, Trace, Artifact
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
parser.add_argument('--learningRate', help='learning rate', default=0.0001)
parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005)
parser.add_argument('--batchSize', help='training batch size', default=128)
parser.add_argument('--validSize', help='validation set size', default=256)
parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64)
parser.add_argument('--noStandardize', help='do not standardize observations', action='store_true')
parser.add_argument('--resume', help='resume training of the latest artifact', action='store_true')
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

def get_batch(data):
    traces = []
    for i in range(len(data)):
        trace = Trace()
        data_i = data[i]
        obs_shape = data_i['observes']['shape']
        obs_data = data_i['observes']['data']
        obs = util.Tensor(obs_data).view(obs_shape)
        if artifact.standardize:
            obs = util.standardize(obs)
        trace.set_observes(obs)

        for timeStep in range(len(data_i['samples'])):
            samples_timeStep = data_i['samples'][timeStep]

            address = samples_timeStep['sample-address']
            instance = samples_timeStep['sample-instance']
            proposal_type = samples_timeStep['proposal-name']
            value = samples_timeStep['value']
            if type(value) != int:
                util.log_error('Unsupported sample value type: ' + str(type(value)))
            value = util.Tensor([value])
            sample = Sample(address, instance, proposal_type, value)
            if proposal_type == 'discreteminmax':
                sample.proposal_extra_params_min = samples_timeStep['proposal-extra-params'][0]
                sample.proposal_extra_params_max = samples_timeStep['proposal-extra-params'][1]
            else:
                util.log_error('Unsupported proposal distribution type: ' + proposal_type)
            trace.add_sample(sample)

            #update the artifact's one-hot dictionary as needed
            artifact.add_one_hot_address(address)
            artifact.add_one_hot_instance(instance)
            artifact.add_one_hot_proposal_type(proposal_type)

        traces.append(trace)
    return traces

def get_sub_batches(batch):
    sb = {}
    for trace in batch:
        h = hash(str(trace))
        if not h in sb:
            sb[h] = []
        sb[h].append(trace)
    ret = []
    for _, t in sb.items():
        ret.append(t)
    return ret


def request_batch(n):
    util.zmq_send_request(opt.server, {'command':'new-batch', 'command-param':n})

def receive_batch():
    sys.stdout.write('Waiting for new batch...                                 \r')
    sys.stdout.flush()
    data = util.zmq_receive_reply()

    sys.stdout.write('New batch received, processing...                        \r')
    sys.stdout.flush()
    b = get_batch(data)

    sys.stdout.write('New batch received, splitting into sub-batches...        \r')
    sys.stdout.flush()
    bs = get_sub_batches(b)
    sys.stdout.write('                                                         \r')
    sys.stdout.flush()

    return bs

artifact = Artifact()
artifact.cuda = opt.cuda
artifact.one_hot_address_dim = opt.oneHotDim
artifact.one_hot_instance_dim = opt.oneHotDim
artifact.one_hot_proposal_type_dim = 1

if not opt.resume:
    artifact.standardize = not opt.noStandardize

# torch.save(artifact, 'art1')
# artifact = torch.load('art1')
# print(artifact)

request_batch(2)
sb = receive_batch()
print(sb)
