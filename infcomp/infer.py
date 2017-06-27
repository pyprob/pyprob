#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Atilim Gunes Baydin, Tuan Anh Le, Mario Lezcano Casado, Frank Wood
# University of Oxford
# May 2016 -- June 2017
#

import infcomp
from infcomp import util
from infcomp.comm import ProposalReplier
import torch
from torch.autograd import Variable
import argparse
from termcolor import colored
import datetime
import time
import sys
from pprint import pformat
import os
from itertools import zip_longest
import csv
import traceback

def main():
    try:
        parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Inference Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for loading artifacts and saving logs', default='.')
        parser.add_argument('--nth', help='show the nth artifact (-1: last)', type=int, default=-1)
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--debug', help='show debugging information as requests arrive', action='store_true')
        parser.add_argument('--server', help='address and port to bind this inference server', default='tcp://0.0.0.0:6666')
        parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
        parser.add_argument('--saveHistAddress', help='save the histogram of addresses (csv)', type=str)
        parser.add_argument('--saveHistTrace', help='save the histogram of trace lengths (csv)', type=str)
        opt = parser.parse_args()

        if opt.version:
            print(infcomp.__version__)
            quit()

        if not opt.saveHistAddress is None:
            address_histogram = {}
            address_distributions = {}
        if not opt.saveHistTrace is None:
            trace_length_histogram = {}

        time_stamp = util.get_time_stamp()
        util.init_logger('{0}/{1}'.format(opt.dir, 'infcomp-infer-log' + time_stamp))
        util.init(opt, 'Inference Engine')

        with ProposalReplier(opt.server) as replier:
            util.log_print()
            util.log_print(colored('[] Loaded artifact', 'blue', attrs=['bold']))
            util.log_print()

            file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp-artifact'), opt.nth)
            artifact = util.load_artifact(file_name, opt.cuda, opt.device)

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
            # spinner = util.Spinner()
            time_last_new_trace = time.time()
            duration_last_trace = 0
            traces_per_sec = 0
            max_traces_per_sec = 0
            traces_per_sec_str = '-       '
            max_traces_per_sec_str = '-       '
            total_traces = 0
            sys.stdout.write('TPS      │ Max. TPS │ Total traces\n')
            sys.stdout.write('─────────┼──────────┼─────────────\n')
            sys.stdout.flush()

            while True:
                # spinner.spin()
                sys.stdout.write('{0} │ {1} │ {2}      \r'.format(traces_per_sec_str, max_traces_per_sec_str, total_traces))
                sys.stdout.flush()
                replier.receive_request(artifact.standardize)

                if replier.new_trace:
                    total_traces += 1
                    duration_last_trace = max(util.epsilon, time.time() - time_last_new_trace)
                    time_last_new_trace = time.time()
                    traces_per_sec = 1 / duration_last_trace
                    if traces_per_sec > max_traces_per_sec:
                        max_traces_per_sec = traces_per_sec
                        max_traces_per_sec_str = '{:8}'.format('{:,}'.format(int(max_traces_per_sec)))
                    if traces_per_sec < 1:
                        traces_per_sec_str = '-       '
                    else:
                        traces_per_sec_str = '{:8}'.format('{:,}'.format(int(traces_per_sec)))

                    # print(duration_last_trace)

                    if not opt.saveHistTrace is None:
                        if time_step != 0:
                            if time_step in trace_length_histogram:
                                trace_length_histogram[time_step] += 1
                            else:
                                trace_length_histogram[time_step] = 1

                    time_step = 0
                    obs = replier.observes
                    observe_embedding = artifact.observe_layer.forward(Variable(obs.unsqueeze(0), volatile=True))
                    replier.reply_observes_received()

                    if opt.debug:
                        util.log_print('ObservesInitRequest')
                        util.log_print('Time: reset')
                        util.log_print('observes: {0}'.format(str(obs.size())))
                        util.log_print()

                else:
                    current_sample = replier.current_sample
                    current_address = current_sample.address_suffixed
                    # current_instance = current_sample.instance
                    current_distribution = current_sample.distribution

                    prev_sample = replier.previous_sample
                    prev_address = prev_sample.address_suffixed
                    # prev_instance = prev_sample.instance
                    prev_distribution = prev_sample.distribution

                    if not opt.saveHistAddress is None:
                        if current_address in address_histogram:
                            address_histogram[current_address] += 1
                        else:
                            address_histogram[current_address] = 1
                        if current_address not in address_distributions:
                            address_distributions[current_address] = current_distribution.name

                    success = True
                    if not current_address in artifact.proposal_layers:
                        util.log_warning('No proposal layer for: {0}'.format(current_address))
                        success = False
                    if current_address in artifact.one_hot_address:
                        current_one_hot_address = artifact.one_hot_address[current_address]
                    else:
                        util.log_warning('Unknown address (current): {0}'.format(current_address))
                        success = False
                    if current_distribution.name in artifact.one_hot_distribution:
                        current_one_hot_distribution = artifact.one_hot_distribution[current_distribution.name]
                    else:
                        util.log_warning('Unknown distribution (current): {0}'.format(current_distribution.name))
                        success = False

                    if time_step == 0:
                        prev_sample_embedding = Variable(util.Tensor(1, artifact.smp_emb_dim).zero_(), volatile=True)

                        prev_one_hot_address = artifact.one_hot_address_empty
                        prev_one_hot_distribution = artifact.one_hot_distribution_empty
                    else:
                        if prev_address in artifact.sample_layers:
                            prev_sample_embedding = artifact.sample_layers[prev_address](Variable(prev_sample.value.unsqueeze(0), volatile=True))
                        else:
                            util.log_warning('No sample embedding layer for: {0}'.format(prev_address))
                            success = False

                        if prev_address in artifact.one_hot_address:
                            prev_one_hot_address = artifact.one_hot_address[prev_address]
                        else:
                            util.log_warning('Unknown address (previous): {0}'.format(prev_address))
                            success = False
                        if prev_distribution.name in artifact.one_hot_distribution:
                            prev_one_hot_distribution = artifact.one_hot_distribution[prev_distribution.name]
                        else:
                            util.log_warning('Unknown distribution (previous): {0}'.format(prev_distribution.name))
                            success = False

                    if success:
                        t = [observe_embedding[0],
                             prev_sample_embedding[0],
                             prev_one_hot_address,
                             prev_one_hot_distribution,
                             current_one_hot_address,
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
                        success, proposal_output = artifact.proposal_layers[current_address].forward(proposal_input, [current_sample])
                        current_sample.distribution.set_proposalparams(proposal_output[0].data)
                    else:
                        util.log_warning('Proposal will be made from the prior.')
                    replier.reply_proposal(success, current_sample.distribution)


                    if opt.debug:
                        util.log_print('ProposalRequest')
                        util.log_print('Time: {0}'.format(time_step))
                        util.log_print('Previous  address          : {0}, distribution: {1}, value: {2}'.format(prev_address, prev_distribution, prev_sample.value.size()))
                        util.log_print('Current (requested) address: {0}, distribution: {1}'.format(current_address, current_distribution))
                        util.log_print()

                    time_step += 1

    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)

    if not opt.saveHistAddress is None:
        util.log_print('Saving address histogram to file: ' + opt.saveHistAddress)
        with open(opt.saveHistAddress, 'w') as f:
            data_address = []
            data_distribution = []
            data_count = []
            for address in address_histogram:
                data_address.append(address)
                data_distribution.append(address_distributions[address])
                data_count.append(address_histogram[address])
            data = [data_address, data_distribution, data_count]
            writer = csv.writer(f)
            writer.writerow(['address', 'distribution', 'count'])
            for values in zip_longest(*data):
                writer.writerow(values)

    if not opt.saveHistTrace is None:
        util.log_print('Saving trace length histogram to file: ' + opt.saveHistTrace)
        with open(opt.saveHistTrace, 'w') as f:
            data_trace_length = []
            data_count = []
            for trace_length in trace_length_histogram:
                data_trace_length.append(trace_length)
                data_count.append(trace_length_histogram[trace_length])
            data = [data_trace_length, data_count]
            writer = csv.writer(f)
            writer.writerow(['trace_length', 'count'])
            for values in zip_longest(*data):
                writer.writerow(values)
    sys.exit(0)

if __name__ == "__main__":
    main()
