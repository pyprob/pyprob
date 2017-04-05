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
from torch.autograd import Variable
import argparse
from termcolor import colored
import datetime
import sys
from pprint import pformat
import os

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--latest', help='show the latest artifact', action='store_true')
parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int)
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=4, type=int)
parser.add_argument('--debug', help='show debugging information as requests arrive', action='store_true')
parser.add_argument('--server', help='address and port to bind this inference serve', default='tcp://127.0.0.1:6666')
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

if not opt.latest and opt.nth is None:
    parser.print_help()
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.folder, 'artifact-info-log' + time_stamp))
util.init(opt)

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
    if opt.latest:
        opt.nth = -1
    file_name = util.file_starting_with('{0}/{1}'.format(opt.folder, 'compile-artifact'), opt.nth)
    artifact = torch.load(file_name)
    file_size = '{:,}'.format(os.path.getsize(file_name))

    util.log_print()
    util.log_print(colored('█ Loaded artifact', 'blue', attrs=['bold']))
    util.log_print()

    artifact = torch.load(file_name)
    prev_artifact_total_traces = artifact.total_traces
    prev_artifact_total_iterations = artifact.total_iterations
    prev_artifact_total_training_time = artifact.total_training_time

    util.check_versions(artifact)

    file_size = '{:,}'.format(os.path.getsize(file_name))
    util.log_print('File name             : {0}'.format(file_name))
    util.log_print('File size (Bytes)     : {0}'.format(file_size))
    util.log_print(artifact.get_info())
    util.log_print()

    util.log_print()
    util.log_print(colored('█ Inference engine running at ' + opt.server, 'blue', attrs=['bold']))
    util.log_print()

    artifact.eval()

    observe = None
    observe_embedding = None
    time_step = 0
    spinner = util.Spinner()
    while True:
        spinner.spin()
        request = replier.receive_request()
        command = request['command']
        command_param = request['command-param']
        if command == 'observe-init':
            time_step = 0
            obs_shape = command_param['shape']
            obs_data = command_param['data']
            obs = util.Tensor(obs_data).view(obs_shape)

            if artifact.standardize:
                obs = util.standardize(obs)

            observe_embedding = artifact.observe_layer(Variable(obs.unsqueeze(0), requires_grad=False))

            replier.send_reply('observe-received')

            if opt.debug:
                util.log_print('Command       : observe-init')
                util.log_print('Command params: {0}'.format(str(obs.size())))
                util.log_print()
        elif command == 'proposal-params':
            address = command_param['sample-address']
            instance = command_param['sample-instance']
            proposal_type = command_param['proposal-name']
            prev_address = command_param['prev-sample-address']
            prev_instance = command_param['prev-sample-instance']
            prev_value = command_param['prev-sample-value']

            if type(prev_value) == int or type(prev_value) == float:
                prev_value = util.Tensor([prev_value])
            else:
                util.log_error('Unsupported sample type: {0}'.format(str(prev_value)))
                quit()

            if opt.debug:
                util.log_print('Command       : proposal-params')
                util.log_print('Command params: requested address: {0}, instance: {1}, proposal type: {2}; previous address: {3}, previous instance: {4}, previous value: {5}'.format(address, instance, proposal_type, prev_address, prev_instance, str(prev_value.size())))
                util.log_print()

            if time_step == 0:
                prev_sample_embedding = Variable(util.Tensor(1, artifact.smp_emb_dim).fill_(0), requires_grad=False)
            else:
                if not (prev_address, prev_instance) in artifact.sample_layers:
                    util.log_error('Artifact has no sample embedding layer for: {0}, {1}'.format(prev_address, prev_instance))
                    quit()
                prev_sample_embedding = artifact.sample_layers[(prev_address, prev_instance)](Variable(prev_value.unsqueeze(0), requires_grad=False))

            t = [observe_embedding[0],
                 prev_sample_embedding[0],
                 artifact.one_hot_address[address],
                 artifact.one_hot_instance[instance],
                 artifact.one_hot_proposal_type[proposal_type]]
            t = torch.cat(t).unsqueeze(0)
            lstm_input = t.unsqueeze(0)

            lstm_output, _ = artifact.lstm(lstm_input)

            if not (address, instance) in artifact.proposal_layers:
                util.log_error('Artifact has no proposal layer for: {0}, {1}'.format(address, instance))
                quit()

            proposal_input = lstm_output[0]
            proposal_output = artifact.proposal_layers[(address, instance)](proposal_input)

            if proposal_type == 'discreteminmax':
                replier.send_reply(proposal_output[0].data.numpy().tolist())
            else:
                util.log_error('Unsupported proposal type: {0}'.format(proposal_type))
                quit()
            time_step += 1

        else:
            util.log_error('Unkown command: {0}'.format(str(request)))
