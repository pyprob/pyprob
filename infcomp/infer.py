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
import traceback

def main():
    try:
        parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory to save artifacts and logs', default='.')
        parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int, default=-1)
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--debug', help='show debugging information as requests arrive', action='store_true')
        parser.add_argument('--server', help='address and port to bind this inference server', default='tcp://127.0.0.1:6666')
        opt = parser.parse_args()

        if opt.version:
            print(infcomp.__version__)
            quit()

        time_stamp = util.get_time_stamp()
        util.init_logger('{0}/{1}'.format(opt.dir, 'infcomp-infer-log' + time_stamp))
        util.init(opt)

        util.log_print()
        util.log_print(colored('[] Oxford Inference Compilation ' + infcomp.__version__, 'blue', attrs=['bold']))
        util.log_print()
        util.log_print('Inference Engine')
        util.log_print()
        util.log_configuration(opt)

        with ProposalReplier(opt.server) as replier:
            util.log_print()
            util.log_print(colored('[] Loaded artifact', 'blue', attrs=['bold']))
            util.log_print()

            file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp-artifact'), opt.nth)
            artifact = util.load_artifact(file_name, opt.cuda)

            prev_artifact_total_traces = artifact.total_traces
            prev_artifact_total_iterations = artifact.total_iterations
            prev_artifact_total_training_seconds = artifact.total_training_seconds

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
                        util.log_print('Time: reset')
                        util.log_print('observes: {0}'.format(str(obs.size())))
                        util.log_print()
                else:
                    current_sample = replier.current_sample
                    current_address = current_sample.address
                    current_instance = current_sample.instance
                    current_distribution = current_sample.distribution

                    prev_sample = replier.previous_sample
                    prev_address = prev_sample.address
                    prev_instance = prev_sample.instance
                    prev_distribution = prev_sample.distribution

                    if opt.debug:
                        util.log_print('ProposalRequest')
                        util.log_print('Time: {0}'.format(time_step))
                        util.log_print('Previous  address          : {0}, instance: {1}, distribution: {2}, value: {3}'.format(prev_address, prev_instance, prev_distribution, prev_sample.value.size()))
                        util.log_print('Current (requested) address: {0}, instance: {1}, distribution: {2}'.format(current_address, current_instance, current_distribution))
                        util.log_print()

                    success = True
                    if not (current_address, current_instance) in artifact.proposal_layers:
                        util.log_warning('No proposal layer for: {0}, {1}'.format(current_address, current_instance))
                        success = False
                    if current_address in artifact.one_hot_address:
                        current_one_hot_address = artifact.one_hot_address[current_address]
                    else:
                        util.log_warning('Unknown address (current): {0}'.format(current_address))
                        success = False
                    if current_instance in artifact.one_hot_instance:
                        current_one_hot_instance = artifact.one_hot_instance[current_instance]
                    else:
                        util.log_warning('Unknown instance (current): {0}'.format(current_instance))
                        success = False
                    if current_distribution.name() in artifact.one_hot_distribution:
                        current_one_hot_distribution = artifact.one_hot_distribution[current_distribution.name()]
                    else:
                        util.log_warning('Unknown distribution (current): {0}'.format(current_distribution.name()))
                        success = False

                    if time_step == 0:
                        prev_sample_embedding = Variable(util.Tensor(1, artifact.smp_emb_dim).zero_(), volatile=True)

                        prev_one_hot_address = artifact.one_hot_address_empty
                        prev_one_hot_instance = artifact.one_hot_instance_empty
                        prev_one_hot_distribution = artifact.one_hot_distribution_empty
                    else:
                        if (prev_address, prev_instance) in artifact.sample_layers:
                            prev_sample_embedding = artifact.sample_layers[(prev_address, prev_instance)](Variable(prev_sample.value.unsqueeze(0), volatile=True))
                        else:
                            util.log_warning('No sample embedding layer for: {0}, {1}'.format(prev_address, prev_instance))
                            success = False

                        if prev_address in artifact.one_hot_address:
                            prev_one_hot_address = artifact.one_hot_address[prev_address]
                        else:
                            util.log_warning('Unknown address (previous): {0}'.format(prev_address))
                            success = False
                        if prev_instance in artifact.one_hot_instance:
                            prev_one_hot_instance = artifact.one_hot_instance[prev_instance]
                        else:
                            util.log_warning('Unknown instance (previous): {0}'.format(prev_instance))
                            success = False
                        if prev_distribution.name() in artifact.one_hot_distribution:
                            prev_one_hot_distribution = artifact.one_hot_distribution[prev_distribution.name()]
                        else:
                            util.log_warning('Unknown distribution (previous): {0}'.format(prev_distribution.name()))
                            success = False

                    if success:
                        t = [observe_embedding[0],
                             prev_sample_embedding[0],
                             prev_one_hot_address,
                             prev_one_hot_instance,
                             prev_one_hot_distribution,
                             current_one_hot_address,
                             current_one_hot_instance,
                             current_one_hot_distribution]
                        t = torch.cat(t).unsqueeze(0)
                        lstm_input = t.unsqueeze(0)

                        if time_step == 0:
                            h0 = Variable(util.Tensor(artifact.lstm_depth, 1, artifact.lstm_dim).zero_(), volatile=True)
                            lstm_hidden_state = (h0, h0)
                            lstm_output, lstm_hidden_state = artifact.lstm(lstm_input, lstm_hidden_state)
                        else:
                            lstm_output, lstm_hidden_state = artifact.lstm(lstm_input, lstm_hidden_state)

                        proposal_input = lstm_output[0]
                        proposal_output = artifact.proposal_layers[(current_address, current_instance)](proposal_input)
                        current_sample.distribution.set_proposalparams(proposal_output[0].data)
                    else:
                        util.log_warning('Proposal will be made from the prior.')

                    replier.reply_proposal(success, current_sample.distribution)

                    time_step += 1
    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
