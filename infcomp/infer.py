#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import infcomp
from infcomp import util
from infcomp.protocol import ProposalReplier
import infcomp.flatbuffers.ObservesInitRequest
import infcomp.flatbuffers.ProposalRequest
import torch
from torch.autograd import Variable
import argparse
from termcolor import colored
import datetime
import sys
from pprint import pformat
import os

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--dir', help='directory to save artifacts and logs', default='.')
parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int, default=-1)
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=4, type=int)
parser.add_argument('--debug', help='show debugging information as requests arrive', action='store_true')
parser.add_argument('--server', help='address and port to bind this inference server', default='tcp://127.0.0.1:6666')
opt = parser.parse_args()

if opt.version:
    print(infcomp.__version__)
    quit()

time_stamp = util.get_time_stamp()
util.init_logger('{0}/{1}'.format(opt.dir, 'artifact-info-log' + time_stamp))
util.init(opt)

util.log_print()
util.log_print(colored('[] Oxford Inference Compilation ' + infcomp.__version__, 'blue', attrs=['bold']))
util.log_print()
util.log_print('Inference Engine')
util.log_print()
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print()
util.log_print('Running on PyTorch ' + torch.__version__)
util.log_print()
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))

util.log_print()
util.log_print(colored('[] Inference configuration', 'blue', attrs=['bold']))
util.log_print()
util.log_print(pformat(vars(opt)))
util.log_print()

def main():
    try:
        with ProposalReplier(opt.server) as replier:
            file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'compile-artifact'), opt.nth)
            artifact = torch.load(file_name)
            file_size = '{:,}'.format(os.path.getsize(file_name))

            util.log_print()
            util.log_print(colored('[] Loaded artifact', 'blue', attrs=['bold']))
            util.log_print()

            artifact = torch.load(file_name)
            prev_artifact_total_traces = artifact.total_traces
            prev_artifact_total_iterations = artifact.total_iterations
            prev_artifact_total_training_seconds = artifact.total_training_seconds

            util.check_versions(artifact)

            file_size = '{:,}'.format(os.path.getsize(file_name))
            util.log_print('File name             : {0}'.format(file_name))
            util.log_print('File size (Bytes)     : {0}'.format(file_size))
            util.log_print(artifact.get_info())
            util.log_print()

            util.log_print()
            util.log_print(colored('[] Inference engine running at ' + opt.server, 'blue', attrs=['bold']))
            util.log_print()

            artifact.eval()

            observe = None
            observe_embedding = None
            lstm_hidden_state = None
            time_step = 0
            spinner = util.Spinner()
            while True:
                spinner.spin()
                replier.receive_request(artifact.standardize)

                if replier.new_trace:
                    time_step = 0
                    obs = replier.observes
                    observe_embedding = artifact.observe_layer(Variable(obs.unsqueeze(0), volatile=True))
                    replier.reply_observes_received()

                    if opt.debug:
                        util.log_print('ObservesInitRequest')
                        util.log_print('observes: {0}'.format(str(obs.size())))
                        util.log_print()
                else:
                    current_sample = replier.current_sample
                    previous_sample = replier.previous_sample

                    if opt.debug:
                        util.log_print('ProposalRequest')
                        util.log_print('requested address: {0}, instance: {1}'.format(current_sample.address, current_sample.instance))
                        util.log_print('previous  address: {0}, instance: {1}, value: {2}'.format(previous_sample.address, previous_sample.instance, previous_sample.value.size()))
                        util.log_print()

                    if time_step == 0:
                        previous_sample_embedding = Variable(util.Tensor(1, artifact.smp_emb_dim).zero_(), volatile=True)
                    else:
                        if not (previous_sample.address, previous_sample.instance) in artifact.sample_layers:
                            util.log_error('Artifact has no sample embedding layer for: {0}, {1}'.format(previous_sample.address, previous_sample.instance))

                        previous_sample_embedding = artifact.sample_layers[(previous_sample.address, previous_sample.instance)](Variable(previous_sample.value.unsqueeze(0), volatile=True))

                    t = [observe_embedding[0],
                         previous_sample_embedding[0],
                         artifact.one_hot_address[current_sample.address],
                         artifact.one_hot_instance[current_sample.instance],
                         artifact.one_hot_distribution[current_sample.distribution.name()]]
                    t = torch.cat(t).unsqueeze(0)
                    lstm_input = t.unsqueeze(0)

                    if time_step == 0:
                        h0 = Variable(util.Tensor(artifact.lstm_depth, 1, artifact.lstm_dim).zero_(), volatile=True)
                        lstm_hidden_state = (h0, h0)
                        lstm_output, lstm_hidden_state = artifact.lstm(lstm_input, lstm_hidden_state)
                    else:
                        lstm_output, lstm_hidden_state = artifact.lstm(lstm_input, lstm_hidden_state)

                    if not (current_sample.address, current_sample.instance) in artifact.proposal_layers:
                        util.log_error('Artifact has no proposal layer for: {0}, {1}'.format(current_sample.address, current_sample.instance))

                    proposal_input = lstm_output[0]
                    proposal_output = artifact.proposal_layers[(current_sample.address, current_sample.instance)](proposal_input)
                    current_sample.distribution.set_proposalparams(proposal_output[0].data)
                    replier.reply_proposal(current_sample.distribution)
                    time_step += 1
    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
