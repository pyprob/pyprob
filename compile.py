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
import sys
import datetime
from pprint import pformat
import os

parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + util.version + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--version', help='show version information', action='store_true')
parser.add_argument('--folder', help='folder to save artifacts and logs', default='./artifacts')
parser.add_argument('--cuda', help='use CUDA', action='store_true')
parser.add_argument('--seed', help='random seed', default=4, type=int)
parser.add_argument('--server', help='address of the probprog model server', default='tcp://127.0.0.1:5555')
parser.add_argument('--learningRate', help='learning rate', default=0.0001, type=float)
parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005, type=float)
parser.add_argument('--batchSize', help='training batch size', default=128, type=int)
parser.add_argument('--validSize', help='validation set size', default=256, type=int)
parser.add_argument('--validInterval', help='validation interval (traces)', default=500, type=int)
parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64, type=int)
parser.add_argument('--standardize', help='standardize observations', action='store_true')
parser.add_argument('--resumeLatest', help='resume training of the latest artifact', action='store_true')
parser.add_argument('--obsEmb', help='observation embedding', choices=['fc', 'cnn6_2d'], default='fc', type=str)
parser.add_argument('--obsEmbDim', help='observation embedding dimension', default=128, type=int)
parser.add_argument('--smpEmb', help='sample embedding', choices=['fc'], default='fc', type=str)
parser.add_argument('--smpEmbDim', help='sample embedding dimension', default=1, type=int)
parser.add_argument('--lstmDim', help='lstm hidden unit dimension', default=256, type=int)
parser.add_argument('--lstmDepth', help='number of stacked lstms', default=2, type=int)
parser.add_argument('--softmaxBoost', help='multiplier before softmax', default=20.0, type=float)
parser.add_argument('--keepArtifacts', help='keep all previously best artifacts during training, do not overwrite', action='store_true')
opt = parser.parse_args()

if opt.version:
    print(util.version)
    quit()

time_stamp = util.get_time_stamp()
artifact_file = '{0}/{1}'.format(opt.folder, 'compile-artifact' + time_stamp)
util.init_logger('{0}/{1}'.format(opt.folder, 'compile-log' + time_stamp))
util.init(opt)

util.log_print()
util.log_print(colored('█ Oxford Inference Compilation ' + util.version, 'blue', attrs=['bold']))
util.log_print()
util.log_print('Compilation Mode')
util.log_print()
util.log_print('Started ' +  str(datetime.datetime.now()))
util.log_print()
util.log_print('Running on PyTorch ' + torch.__version__)
util.log_print()
util.log_print('Command line arguments:')
util.log_print(' '.join(sys.argv[1:]))

util.log_print()
util.log_print(colored('█ Compilation configuration', 'blue', attrs=['bold']))
util.log_print()
util.log_print(pformat(vars(opt)))
util.log_print()

with Requester(opt.server) as requester:
    if opt.resumeLatest:
        resume_artifact_file = util.file_starting_with('{0}/{1}'.format(opt.folder, 'compile-artifact'), -1)
        util.log_print()
        util.log_print(colored('█ Resuming artifact', 'blue', attrs=['bold']))
        util.log_print()

        artifact = torch.load(resume_artifact_file)
        prev_artifact_total_traces = artifact.total_traces
        prev_artifact_total_iterations = artifact.total_iterations
        prev_artifact_total_training_time = artifact.total_training_time

        util.check_versions(artifact)

        file_size = '{:,}'.format(os.path.getsize(resume_artifact_file))
        util.log_print('File name             : {0}'.format(resume_artifact_file))
        util.log_print('File size (Bytes)     : {0}'.format(file_size))
        util.log_print(artifact.get_info())
        util.log_print()
        util.log_print('New artifact will be saved to: ' + artifact_file)
    else:
        util.log_print()
        util.log_print(colored('█ New artifact', 'blue', attrs=['bold']))
        util.log_print()
        util.log_print('File name: ' + artifact_file)

        prev_artifact_total_traces = 0
        prev_artifact_total_iterations = 0
        prev_artifact_total_training_time = datetime.timedelta(0)

        artifact = Artifact()
        artifact.on_cuda = opt.cuda
        artifact.standardize = opt.standardize
        artifact.one_hot_address_dim = opt.oneHotDim
        artifact.one_hot_instance_dim = opt.oneHotDim
        artifact.one_hot_proposal_type_dim = 1
        artifact.valid_size = opt.validSize
        requester.request_batch(artifact.valid_size)
        artifact.valid_batch = requester.receive_batch(artifact.standardize)

        example_observes = artifact.valid_batch[0][0].observes
        artifact.set_observe_embedding(example_observes, opt.obsEmb, opt.obsEmbDim)
        artifact.set_sample_embedding(opt.smpEmb, opt.smpEmbDim)
        artifact.set_lstm(opt.lstmDim, opt.lstmDepth)

        artifact.softmax_boost = opt.softmaxBoost

        artifact.polymorph()
        if opt.cuda:
            artifact.cuda()

    optimizer = optim.Adam(artifact.parameters(), lr=opt.learningRate, weight_decay=opt.weightDecay)
    if not artifact.optimizer_state is None:
        optimizer.load_state_dict(artifact.optimizer_state)

    iteration = 0
    trace = 0
    start_time = datetime.datetime.now()
    improvement_time = datetime.datetime.now()
    train_loss_str = ''
    if artifact.valid_loss_best is None:
        artifact.valid_loss_best = artifact.valid_loss()
    if artifact.valid_loss_worst is None:
        artifact.valid_loss_worst = artifact.valid_loss_best
    if artifact.valid_loss_initial is None:
        artifact.valid_loss_initial = artifact.valid_loss_best
    if prev_artifact_total_traces == 0:
        artifact.valid_history_trace.append(prev_artifact_total_traces + iteration)
        artifact.valid_history_loss.append(artifact.valid_loss_best)
    valid_loss_best_str = '{:+.6e}'.format(artifact.valid_loss_best)
    valid_loss_str = '{:+.6e}  '.format(artifact.valid_history_loss[-1])
    last_validation_trace = 0

    util.log_print()
    util.log_print(colored('█ Training from ' + opt.server, 'blue', attrs=['bold']))
    util.log_print()

    time_str = util.days_hours_mins_secs(prev_artifact_total_training_time + (datetime.datetime.now() - start_time))
    improvement_time_str = util.days_hours_mins_secs(datetime.datetime.now() - improvement_time)
    trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))
    util.log_print('{{:{0}}}'.format(len(time_str)).format('Train. time') + ' │ ' + '{{:{0}}}'.format(len(trace_str)).format('Trace') + ' │ Training loss   │ Last valid. loss│ Best val. loss|' + '{{:{0}}}'.format(len(improvement_time_str)).format('T.since best'))
    util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────┼───────────────┼─' + '─'*len(improvement_time_str))

    requester.request_batch(opt.batchSize)
    while True:
        batch = requester.receive_batch(artifact.standardize)
        requester.request_batch(opt.batchSize)
        artifact.polymorph(batch)

        for sub_batch in batch:
            iteration += 1
            sys.stdout.write('Training...                                              \r')
            sys.stdout.flush()

            artifact.train()
            optimizer.zero_grad()
            loss = artifact.loss(sub_batch)
            loss.backward()
            optimizer.step()
            train_loss = loss.data[0]

            trace += len(sub_batch)

            artifact.total_training_time = prev_artifact_total_training_time + (datetime.datetime.now() - start_time)
            artifact.total_iterations = prev_artifact_total_iterations + iteration
            artifact.total_traces = prev_artifact_total_traces + trace

            artifact.train_history_trace.append(artifact.total_traces)
            artifact.train_history_loss.append(train_loss)

            if train_loss < artifact.train_loss_best:
                artifact.train_loss_best = train_loss
                train_loss_str = colored('{:+.6e} ▼'.format(train_loss), 'green', attrs=['bold'])
            elif train_loss > artifact.train_loss_worst:
                artifact.train_loss_worst = train_loss
                train_loss_str = colored('{:+.6e} ▲'.format(train_loss), 'red', attrs=['bold'])
            elif train_loss < artifact.valid_history_loss[-1]:
                train_loss_str = colored('{:+.6e}  '.format(train_loss), 'green')
            elif train_loss > artifact.valid_history_loss[-1]:
                train_loss_str = colored('{:+.6e}  '.format(train_loss), 'red')
            else:
                train_loss_str = '{:+.6e}  '.format(train_loss)

            time_str = util.days_hours_mins_secs(prev_artifact_total_training_time + (datetime.datetime.now() - start_time))
            trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))

            if trace - last_validation_trace > opt.validInterval:
                util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────┼───────────────┼─' + '─'*len(improvement_time_str))
                sys.stdout.write('Computing validation loss...                             \r')
                sys.stdout.flush()

                artifact.eval()
                valid_loss = artifact.valid_loss()
                last_validation_trace = trace - 1

                artifact.valid_history_trace.append(artifact.total_traces)
                artifact.valid_history_loss.append(valid_loss)

                valid_loss_best_str = '{:+.6e}'.format(artifact.valid_loss_best)
                if valid_loss < artifact.valid_loss_best:
                    artifact.valid_loss_best = valid_loss
                    valid_loss_str = colored('{:+.6e} ▼'.format(valid_loss), 'green', attrs=['bold'])
                    valid_loss_best_str = colored('{:+.6e}'.format(artifact.valid_loss_best), 'green', attrs=['bold'])

                    # save artifact here
                    sys.stdout.write('Updating best artifact on disk...                        \r')
                    sys.stdout.flush()
                    artifact.valid_loss_final = valid_loss
                    artifact.modified = datetime.datetime.now()
                    artifact.updates += 1
                    artifact.optimizer_state = optimizer.state_dict()
                    if opt.keepArtifacts:
                        time_stamp = util.get_time_stamp()
                        artifact_file = '{0}/{1}'.format(opt.folder, 'compile-artifact' + time_stamp)
                    torch.save(artifact, artifact_file)

                    improvement_time = datetime.datetime.now()
                elif valid_loss > artifact.valid_loss_worst:
                    artifact.valid_loss_worst = valid_loss
                    valid_loss_str = colored('{:+.6e} ▲'.format(valid_loss), 'red', attrs=['bold'])
                elif valid_loss < artifact.valid_history_loss[-1]:
                    valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'green')
                elif valid_loss > artifact.valid_history_loss[-1]:
                    valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'red')
                else:
                    valid_loss_str = '{:+.6e}  '.format(valid_loss)

            improvement_time_str = util.days_hours_mins_secs(datetime.datetime.now() - improvement_time)
            util.log_print('{0} │ {1} │ {2} │ {3} │ {4} │ {5} '.format(time_str, trace_str, train_loss_str, valid_loss_str, valid_loss_best_str, improvement_time_str))



# print(artifact.loss(artifact.valid_batch[0]))
