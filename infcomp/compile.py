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
from infcomp.protocol import BatchRequester
from infcomp.modules import Artifact

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
import sys
import datetime
import time
import os
import traceback
from threading import Thread

def validate(artifact, opt, optimizer, artifact_file):
    sys.stdout.write('Computing validation loss...                             \r')
    sys.stdout.flush()

    artifact.eval()
    valid_loss = artifact.valid_loss()

    artifact.valid_history_trace.append(artifact.total_traces)
    artifact.valid_history_loss.append(valid_loss)

    valid_loss_best_str = '{:+.6e}'.format(artifact.valid_loss_best)
    improved = False
    if valid_loss < artifact.valid_loss_best:
        artifact.valid_loss_best = valid_loss
        valid_loss_str = colored('{:+.6e} ▼'.format(valid_loss), 'green', attrs=['bold'])
        valid_loss_best_str = colored('{:+.6e}'.format(artifact.valid_loss_best), 'green', attrs=['bold'])

        sys.stdout.write('Updating best artifact on disk...                        \r')
        sys.stdout.flush()
        artifact.valid_loss_final = valid_loss
        artifact.modified = datetime.datetime.now()
        artifact.updates += 1
        artifact.optimizer = opt.optimizer
        artifact.optimizer_state = optimizer.state_dict()
        if opt.keepArtifacts:
            time_stamp = util.get_time_stamp()
            artifact_file = '{0}/{1}'.format(opt.dir, 'infcomp-artifact' + time_stamp)
        def save_artifact():
            torch.save(artifact, artifact_file)
        a = Thread(target=save_artifact)
        a.start()
        a.join()
        improved = True
    elif valid_loss > artifact.valid_loss_worst:
        artifact.valid_loss_worst = valid_loss
        valid_loss_str = colored('{:+.6e} ▲'.format(valid_loss), 'red', attrs=['bold'])
    elif valid_loss < artifact.valid_history_loss[-1]:
        valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'green')
    elif valid_loss > artifact.valid_history_loss[-1]:
        valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'red')
    else:
        valid_loss_str = '{:+.6e}  '.format(valid_loss)
    return improved, (valid_loss_str, valid_loss_best_str)

def main():
    try:
        parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory to save artifacts and logs', default='.')
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--server', help='address of the probprog model server', default='tcp://127.0.0.1:5555')
        parser.add_argument('--optimizer', help='optimizer for training the artifact', choices=['adam', 'sgd'], default='adam', type=str)
        parser.add_argument('--learningRate', help='learning rate', default=0.0001, type=float)
        parser.add_argument('--momentum', help='momentum (only for sgd)', default=0.9, type=float)
        parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005, type=float)
        parser.add_argument('--clip', help='gradient clipping (-1: disabled)', default=-1, type=float)
        parser.add_argument('--batchSize', help='training batch size', default=128, type=int)
        parser.add_argument('--validSize', help='validation set size', default=256, type=int)
        parser.add_argument('--validTraces', help='validation interval (traces)', default=1000, type=int)
        parser.add_argument('--maxTraces', help='stop training after this many traces (-1: disabled)', default=-1, type=int)
        parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64, type=int)
        parser.add_argument('--standardize', help='standardize observations', action='store_true')
        parser.add_argument('--resume', help='resume training of the latest artifact', action='store_true')
        parser.add_argument('--obsEmb', help='observation embedding', choices=['fc', 'cnn6', 'lstm'], default='fc', type=str)
        parser.add_argument('--obsEmbDim', help='observation embedding dimension', default=128, type=int)
        parser.add_argument('--smpEmb', help='sample embedding', choices=['fc'], default='fc', type=str)
        parser.add_argument('--smpEmbDim', help='sample embedding dimension', default=1, type=int)
        parser.add_argument('--lstmDim', help='lstm hidden unit dimension', default=256, type=int)
        parser.add_argument('--lstmDepth', help='number of stacked lstms', default=2, type=int)
        parser.add_argument('--softmaxBoost', help='multiplier before softmax', default=20.0, type=float)
        parser.add_argument('--keepArtifacts', help='keep all previously best artifacts during training, do not overwrite', action='store_true')
        opt = parser.parse_args()

        if opt.version:
            print(infcomp.__version__)
            quit()

        time_stamp = util.get_time_stamp()
        artifact_file = '{0}/{1}'.format(opt.dir, 'infcomp-artifact' + time_stamp)
        util.init_logger('{0}/{1}'.format(opt.dir, 'infcomp-compile-log' + time_stamp))
        util.init(opt, 'Compilation Mode')

        with BatchRequester(opt.server) as requester:
            if opt.resume:
                util.log_print()
                util.log_print(colored('[] Resuming artifact', 'blue', attrs=['bold']))
                util.log_print()

                resume_artifact_file = util.file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp-artifact'), -1)
                artifact = util.load_artifact(resume_artifact_file, opt.cuda, opt.device)

                prev_artifact_total_traces = artifact.total_traces
                prev_artifact_total_iterations = artifact.total_iterations
                prev_artifact_total_training_seconds = artifact.total_training_seconds

                util.log_print('New artifact will be saved to: ' + artifact_file)
            else:
                util.log_print()
                util.log_print(colored('[] New artifact', 'blue', attrs=['bold']))
                util.log_print()
                util.log_print('File name: ' + artifact_file)

                prev_artifact_total_traces = 0
                prev_artifact_total_iterations = 0
                prev_artifact_total_training_seconds = 0

                artifact = Artifact()
                artifact.on_cuda = opt.cuda
                artifact.cuda_device_id = opt.device
                artifact.standardize = opt.standardize
                artifact.set_one_hot_dims(opt.oneHotDim, opt.oneHotDim, 5)
                artifact.valid_size = opt.validSize
                requester.request_batch(artifact.valid_size)
                artifact.valid_batch = requester.receive_batch(artifact.standardize)

                example_observes = artifact.valid_batch[0][0].observes
                artifact.set_observe_embedding(example_observes, opt.obsEmb, opt.obsEmbDim)
                artifact.set_sample_embedding(opt.smpEmb, opt.smpEmbDim)
                artifact.set_lstm(opt.lstmDim, opt.lstmDepth)

                artifact.softmax_boost = opt.softmaxBoost

                artifact.polymorph()

            if opt.optimizer == 'adam':
                optimizer = optim.Adam(artifact.parameters(), lr=opt.learningRate, weight_decay=opt.weightDecay)
            else:
                optimizer = optim.SGD(artifact.parameters(), lr=opt.learningRate, momentum=opt.momentum, weight_decay=opt.weightDecay)

            if (not artifact.optimizer_state is None) and (artifact.optimizer == opt.optimizer):
                optimizer.load_state_dict(artifact.optimizer_state)

            iteration = 0
            trace = 0
            start_time = time.time()
            improvement_time = start_time
            train_loss_str = '               '
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
            util.log_print(colored('[] Training from ' + opt.server, 'blue', attrs=['bold']))
            util.log_print()

            time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - start_time))
            improvement_time_str = util.days_hours_mins_secs(time.time() - improvement_time)
            trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))
            util.log_print('{{:{0}}}'.format(len(time_str)).format('Train. time') + ' │ ' + '{{:{0}}}'.format(len(trace_str)).format('Trace') + ' │ Training loss   │ Last valid. loss│ Best val. loss|' + '{{:{0}}}'.format(len(improvement_time_str)).format('T.since best'))
            util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────┼───────────────┼─' + '─'*len(improvement_time_str))

            stop = False
            requester.request_batch(opt.batchSize)
            while not stop:
                batch = requester.receive_batch(artifact.standardize)
                requester.request_batch(opt.batchSize)
                artifact.polymorph(batch)

                for sub_batch in batch:
                    iteration += 1
                    sys.stdout.write('Training...                                              \r')
                    sys.stdout.flush()

                    artifact.train()
                    optimizer.zero_grad()
                    art = artifact
                    if opt.cuda:
                        art = torch.nn.DataParallel(artifact)
                    loss = art(sub_batch)
                    loss.backward()
                    if opt.clip > 0:
                        torch.nn.utils.clip_grad_norm(artifact.parameters(), opt.clip)

                    optimizer.step()
                    train_loss = loss.data[0]

                    trace += len(sub_batch)
                    if opt.maxTraces != -1:
                        if trace >= opt.maxTraces:
                            stop = True

                    artifact.total_training_seconds = prev_artifact_total_training_seconds + (time.time() - start_time)
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

                    time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - start_time))
                    trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))

                    if (trace - last_validation_trace > opt.validTraces) or stop:
                        util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────┼───────────────┼─' + '─'*len(improvement_time_str))
                        improved, (valid_loss_str, valid_loss_best_str) = validate(artifact, opt, optimizer, artifact_file)
                        if improved:
                            improvement_time = time.time()
                        last_validation_trace = trace - 1
                    improvement_time_str = util.days_hours_mins_secs(time.time() - improvement_time)
                    util.log_print('{0} │ {1} │ {2} │ {3} │ {4} │ {5} '.format(time_str, trace_str, train_loss_str, valid_loss_str, valid_loss_best_str, improvement_time_str))
            util.log_print('Stopped after {0} traces'.format(trace))
    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
        util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────┼───────────────┼─' + '─'*len(improvement_time_str))
        improved, (valid_loss_str, valid_loss_best_str) = validate(artifact, opt, optimizer, artifact_file)
        if improved:
            improvement_time = time.time()
        last_validation_trace = trace - 1
        improvement_time_str = util.days_hours_mins_secs(time.time() - improvement_time)
        util.log_print('{0} │ {1} │ {2} │ {3} │ {4} │ {5} '.format(time_str, trace_str, train_loss_str, valid_loss_str, valid_loss_best_str, improvement_time_str))
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
