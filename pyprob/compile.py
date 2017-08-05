#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
from pyprob.comm import BatchRequester
from pyprob.modules import Artifact, Batch

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
import numpy as np

def save_artifact(artifact, artifact_file, opt):
    sys.stdout.write('Updating artifact on disk...                             \r')
    sys.stdout.flush()
    artifact.modified = util.get_time_str()
    artifact.updates += 1
    artifact.optimizer = opt.optimizer
    artifact.trained_on = 'CUDA' if opt.cuda else 'CPU'
    if opt.keepArtifacts:
        time_stamp = util.get_time_stamp()
        artifact_file = '{0}/{1}'.format(opt.dir, 'pyprob-artifact' + time_stamp)
    def thread_save():
        torch.save(artifact, artifact_file)
    a = Thread(target=thread_save)
    a.start()
    a.join()

def main():
    try:
        parser = argparse.ArgumentParser(description='pyprob ' + pyprob.__version__ + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for saving artifacts and logs', default='.')
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--parallel', help='parallelize on CUDA using DataParallel', action='store_true')
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--server', help='address of the probprog model server', default='tcp://127.0.0.1:5555')
        parser.add_argument('--optimizer', help='optimizer for training the artifact', choices=['adam', 'sgd'], default='adam', type=str)
        parser.add_argument('--learningRate', help='learning rate', default=0.0001, type=float)
        parser.add_argument('--momentum', help='momentum (only for sgd)', default=0.9, type=float)
        parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005, type=float)
        parser.add_argument('--clip', help='gradient clipping (-1: disabled)', default=-1, type=float)
        parser.add_argument('--batchSize', help='training batch size', default=64, type=int)
        parser.add_argument('--validSize', help='validation set size', default=256, type=int)
        parser.add_argument('--replaceValidBatch', help='replace the validation batch of a resumed artifact', action='store_true')
        parser.add_argument('--validInterval', help='validation interval (traces)', default=1000, type=int)
        parser.add_argument('--maxTraces', help='stop training after this many traces (-1: disabled)', default=-1, type=int)
        parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64, type=int)
        parser.add_argument('--standardize', help='standardize observations', action='store_true')
        parser.add_argument('--resume', help='resume training of the latest artifact', action='store_true')
        parser.add_argument('--obsReshape', help='reshape a 1d observation to a given shape (example: "1x10x10" will reshape 100 -> 1x10x10)', default='', type=str)
        parser.add_argument('--obsEmb', help='observation embedding', choices=['fc', 'cnn1d2c', 'cnn2d6c', 'cnn3d4c', 'lstm'], default='fc', type=str)
        parser.add_argument('--obsEmbDim', help='observation embedding dimension', default=512, type=int)
        parser.add_argument('--smpEmbDim', help='sample embedding dimension', default=32, type=int)
        parser.add_argument('--lstmDim', help='lstm hidden unit dimension', default=512, type=int)
        parser.add_argument('--lstmDepth', help='number of stacked lstms', default=2, type=int)
        parser.add_argument('--dropout', help='dropout value', default=0.2, type=float)
        parser.add_argument('--softmaxBoost', help='multiplier before softmax', default=20.0, type=float)
        parser.add_argument('--keepArtifacts', help='keep all previously best artifacts during training, do not overwrite', action='store_true')
        parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
        parser.add_argument('--batchPool', help='use batches stored in files under the given path (instead of online training with ZMQ)', default='', type=str)
        parser.add_argument('--truncateBackprop', help='use truncated backpropagation through time if sequence length is greater than the given value (-1: disabled)', default=100, type=int)
        opt = parser.parse_args()

        if opt.version:
            print(pyprob.__version__)
            quit()

        time_stamp = util.get_time_stamp()
        artifact_file = '{0}/{1}'.format(opt.dir, 'pyprob-artifact' + time_stamp)
        util.init_logger('{0}/{1}'.format(opt.dir, 'pyprob-compile-log' + time_stamp))
        util.init(opt, 'Compilation Mode')

        if opt.batchPool == '':
            data_source = opt.server
            batch_pool = False
        else:
            data_source = opt.batchPool
            batch_pool = True

        with BatchRequester(data_source, opt.standardize, batch_pool) as requester:
            if opt.resume:
                util.log_print()
                util.log_print(colored('[] Resuming artifact', 'blue', attrs=['bold']))
                util.log_print()

                resume_artifact_file = util.file_starting_with('{0}/{1}'.format(opt.dir, 'pyprob-artifact'), -1)
                artifact = util.load_artifact(resume_artifact_file, opt.cuda, opt.device)
                requester.standardize = artifact.standardize

                if opt.replaceValidBatch:
                    util.log_warning('Replacing the validation batch of the resumed artifact')
                    artifact.valid_size = opt.validSize
                    traces, _ = requester.get_traces(artifact.valid_size, discard_source=True)
                    artifact.valid_batch = Batch(traces)

                prev_artifact_total_traces = artifact.total_traces
                prev_artifact_total_iterations = artifact.total_iterations
                prev_artifact_total_training_seconds = artifact.total_training_seconds

                util.log_print('New artifact will be saved to: ' + artifact_file)

                train_loss_best_str = '{:+.6e}'.format(artifact.train_loss_best)
            else:
                util.log_print()
                util.log_print(colored('[] New artifact', 'blue', attrs=['bold']))
                util.log_print()
                util.log_print('File name: ' + artifact_file)

                prev_artifact_total_traces = 0
                prev_artifact_total_iterations = 0
                prev_artifact_total_training_seconds = 0

                artifact = Artifact()
                artifact.dropout = opt.dropout
                artifact.on_cuda = opt.cuda
                if opt.cuda:
                    artifact.cuda_device_id = torch.cuda.current_device()
                else:
                    artifact.cuda_device_id = opt.device
                artifact.standardize = opt.standardize
                artifact.set_one_hot_dims(opt.oneHotDim, 10)
                artifact.valid_size = opt.validSize
                traces, _ = requester.get_traces(artifact.valid_size, discard_source=True)
                artifact.valid_batch = Batch(traces)

                example_observes = artifact.valid_batch[0].observes
                if opt.obsReshape != '':
                    try:
                        obs_reshape = [int(x) for x in opt.obsReshape.split('x')]
                        reshape_test = example_observes.view(obs_reshape)
                    except:
                        util.log_error('Invalid obsReshape argument. Expecting a format where dimensions are separated by "x" (example: "1x10x10"). The total number of elements in the original 1d input and the requested shape should be the same (example: 100 -> "1x10x10" or "2x50").')
                    artifact.set_observe_embedding(example_observes, opt.obsEmb, opt.obsEmbDim, obs_reshape)
                else:
                    artifact.set_observe_embedding(example_observes, opt.obsEmb, opt.obsEmbDim)
                artifact.set_sample_embedding(opt.smpEmbDim)
                artifact.set_lstm(opt.lstmDim, opt.lstmDepth)

                artifact.softmax_boost = opt.softmaxBoost

                artifact.polymorph()
                artifact.total_traces = 0
                train_loss_best_str = '             '

            if opt.optimizer == 'adam':
                optimizer = optim.Adam(artifact.parameters(), lr=opt.learningRate, weight_decay=opt.weightDecay)
            else:
                optimizer = optim.SGD(artifact.parameters(), lr=opt.learningRate, momentum=opt.momentum, weight_decay=opt.weightDecay)

            iteration = 0
            iteration_batch = 0
            trace = 0
            time_start = time.time()
            time_improvement = time_start
            time_last_batch = time_start
            time_spent_validation = -1
            # train_loss_str = '               '

            if artifact.valid_loss_best is None:
                artifact.valid_loss_best = artifact.valid_loss(opt.parallel)
            if artifact.valid_loss_worst is None:
                artifact.valid_loss_worst = artifact.valid_loss_best
            if prev_artifact_total_traces == 0:
                artifact.valid_history_trace.append(prev_artifact_total_traces + iteration)
                artifact.valid_history_loss.append(artifact.valid_loss_best)
            valid_loss_str = '{:+.6e}  '.format(artifact.valid_history_loss[-1])
            last_validation_trace = 0

            util.log_print()
            util.log_print(colored('[] Training from ' + opt.server, 'blue', attrs=['bold']))
            util.log_print()

            if opt.visdom:
                visdom_panes = {}
                if len(artifact.train_history_trace) == 0:
                    x = torch.zeros(1)
                    y = torch.Tensor([artifact.valid_history_loss[-1]])
                else:
                    x = torch.Tensor(artifact.train_history_trace)
                    y = torch.Tensor(artifact.train_history_loss)
                visdom_panes['train_loss'] = util.vis.line(X=x, Y=y, opts=dict(title='Training loss', xlabel='Trace', ylabel='Loss'))
                if len(artifact.valid_history_trace) == 0:
                    x = torch.zeros(1)
                    y = torch.Tensor([artifact.valid_history_loss[-1]])
                else:
                    x = torch.Tensor(artifact.valid_history_trace)
                    y = torch.Tensor(artifact.valid_history_loss)
                visdom_panes['valid_loss'] = util.vis.line(X=x, Y=y, opts=dict(title='Validation loss', xlabel='Trace', ylabel='Loss'))
                visdom_panes['trace_lengths'] = util.vis.histogram(torch.zeros(2), opts=dict(title='Trace length', numbins=10))
                visdom_panes['performance'] = util.vis.line(X=torch.zeros(2),Y=torch.zeros(2), opts=dict(xlabel='Minibatch', ylabel='Traces / s', title='Performance'))
                visdom_panes['minibatch_time'] = vis_time = util.vis.line(X=torch.zeros(2),Y=torch.zeros(2), opts=dict(xlabel='Minibatch', ylabel='ms', title='Waiting time for minibatch'))
                visdom_panes['addresses'] = util.vis.text(', '.join(list(artifact.one_hot_address.keys())), opts=dict(title='Addresses'))
                visdom_panes['distributions'] = util.vis.text(', '.join(list(artifact.one_hot_distribution.keys())), opts=dict(title='Distributions'))
                visdom_panes['parameters'] = util.vis.line(X=torch.Tensor([0, 1]),Y=torch.Tensor([artifact.num_params_history_num_params[-1] / 1e6, artifact.num_params_history_num_params[-1] / 1e6]), opts=dict(xlabel='Minibatch', ylabel='M', title='Number of parameters'))
                for m_name, m in artifact.named_modules():
                    for p_name, p in m.named_parameters():
                        name = m_name + '_' + p_name + '_' + str(list(p.size()))
                        if name[0] == '_' and not 'bias' in name:
                            visdom_panes[name] =  util.vis.image(util.weights_to_visdom_image(p), opts=dict(caption=name))

            time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - time_start))
            time_improvement_str = util.days_hours_mins_secs(time.time() - time_improvement)
            trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))
            traces_per_sec_str = '   '
            util.log_print('{{:{0}}}'.format(len(time_str)).format('Train. time') + ' │ ' + '{{:{0}}}'.format(len(trace_str)).format('Trace') + ' │ Training loss   │ Min.train.loss│ Valid. loss     |' + '{{:{0}}}'.format(len(time_improvement_str)).format('T.since best') + ' │ TPS')
            util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼───────────────┼─────────────────┼─' + '─'*len(time_improvement_str) + '─┼─' + '─'*len(traces_per_sec_str))

            stop = False
            while not stop:
                save_new_artifact = False
                iteration_batch += 1
                traces, time_wait = requester.get_traces(opt.batchSize)
                batch = Batch(traces)
                if artifact.polymorph(batch):
                    if opt.visdom:
                        util.vis.text(', '.join(list(artifact.one_hot_address.keys())), win=visdom_panes['addresses'])
                        util.vis.text(', '.join(list(artifact.one_hot_distribution.keys())), win=visdom_panes['distributions'])
                        util.vis.line(X=torch.Tensor([iteration_batch]).unsqueeze(0),Y=torch.Tensor([artifact.num_params_history_num_params[-1] / 1e6]).unsqueeze(0), win=visdom_panes['parameters'], update='append')

                # Time statistics
                time_spent_last_batch = max(util.epsilon, time.time() - time_last_batch)
                if time_spent_validation != -1:
                    time_spent_last_batch -= time_spent_validation
                time_last_batch = time.time()
                traces_per_sec = opt.batchSize / time_spent_last_batch

                if opt.visdom:
                    if iteration_batch % 25 == 0:
                        for m_name, m in artifact.named_modules():
                            for p_name, p in m.named_parameters():
                                name = m_name + '_' + p_name + '_' + str(list(p.size()))
                                if name[0] == '_' and not 'bias' in name:
                                    util.vis.image(util.weights_to_visdom_image(p), win=visdom_panes[name], opts=dict(caption=name))
                    tl = util.get_trace_lengths(batch)
                    if len(tl) < 2:
                        tl.append(tl[-1]) # Temporary, due to a bug in Visdom
                    util.vis.histogram(torch.Tensor(tl), win=visdom_panes['trace_lengths'])
                    if trace > 0:
                        util.vis.line(X=torch.Tensor([iteration_batch]).unsqueeze(0),Y=torch.Tensor([time_wait * 1000]).unsqueeze(0), win=visdom_panes['minibatch_time'], update='append')
                        util.vis.line(X=torch.Tensor([iteration_batch]).unsqueeze(0),Y=torch.Tensor([traces_per_sec]).unsqueeze(0), win=visdom_panes['performance'], update='append')

                iteration += 1
                sys.stdout.write('Training...                                              \r')
                sys.stdout.flush()

                artifact.train()
                loss = artifact.loss(batch, optimizer=optimizer, truncate=opt.truncateBackprop, grad_clip=opt.clip, data_parallel=opt.parallel)
                train_loss = loss.data[0]

                trace += batch.length
                if opt.maxTraces != -1:
                    if trace >= opt.maxTraces:
                        stop = True

                artifact.total_training_seconds = prev_artifact_total_training_seconds + (time.time() - time_start)
                artifact.total_iterations = prev_artifact_total_iterations + iteration
                artifact.total_traces = prev_artifact_total_traces + trace

                artifact.train_history_trace.append(artifact.total_traces)
                artifact.train_history_loss.append(train_loss)

                # Compute validation loss as needed
                time_spent_validation = -1
                if (trace - last_validation_trace > opt.validInterval) or stop:
                    time_validation_start = time.time()
                    util.log_print('─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼───────────────┼─────────────────┼─' + '─'*len(time_improvement_str) + '─┼─' + '─'*len(traces_per_sec_str))

                    save_new_artifact = True

                    sys.stdout.write('Computing validation loss...                             \r')
                    sys.stdout.flush()

                    artifact.eval()
                    valid_loss = artifact.valid_loss(opt.parallel)

                    artifact.valid_history_trace.append(artifact.total_traces)
                    artifact.valid_history_loss.append(valid_loss)

                    if valid_loss < artifact.valid_loss_best:
                        artifact.valid_loss_best = valid_loss
                        valid_loss_str = colored('{:+.6e} ▼'.format(valid_loss), 'green', attrs=['bold'])
                    elif valid_loss > artifact.valid_loss_worst:
                        artifact.valid_loss_worst = valid_loss
                        valid_loss_str = colored('{:+.6e} ▲'.format(valid_loss), 'red', attrs=['bold'])
                    elif valid_loss < artifact.valid_history_loss[-1]:
                        valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'green')
                    elif valid_loss > artifact.valid_history_loss[-1]:
                        valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'red')
                    else:
                        valid_loss_str = '{:+.6e}  '.format(valid_loss)

                    last_validation_trace = trace - 1
                    if opt.visdom:
                        x = torch.Tensor([artifact.valid_history_trace[-1]])
                        y = torch.Tensor([artifact.valid_history_loss[-1]])
                        util.vis.line(X=x, Y=y, win=visdom_panes['valid_loss'], update='append')
                    time_spent_validation = time.time() - time_validation_start

                    train_loss_best_str = '{:+.6e}'.format(artifact.train_loss_best)

                if opt.visdom:
                    x = torch.Tensor([artifact.train_history_trace[-1]])
                    y = torch.Tensor([artifact.train_history_loss[-1]])
                    util.vis.line(X=x, Y=y, win=visdom_panes['train_loss'], update='append')

                if train_loss < artifact.train_loss_best:
                    artifact.train_loss_best = train_loss
                    train_loss_str = colored('{:+.6e} ▼'.format(train_loss), 'green', attrs=['bold'])
                    train_loss_best_str = colored('{:+.6e}'.format(artifact.train_loss_best), 'green', attrs=['bold'])

                    save_new_artifact = True

                    time_improvement = time.time()
                elif train_loss > artifact.train_loss_worst:
                    artifact.train_loss_worst = train_loss
                    train_loss_str = colored('{:+.6e} ▲'.format(train_loss), 'red', attrs=['bold'])
                elif train_loss < artifact.valid_history_loss[-1]:
                    train_loss_str = colored('{:+.6e}  '.format(train_loss), 'green')
                elif train_loss > artifact.valid_history_loss[-1]:
                    train_loss_str = colored('{:+.6e}  '.format(train_loss), 'red')
                else:
                    train_loss_str = '{:+.6e}  '.format(train_loss)

                time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - time_start))
                trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))
                traces_per_sec_str = '{:3}'.format('{:,}'.format(int(traces_per_sec)))

                time_improvement_str = util.days_hours_mins_secs(time.time() - time_improvement)
                util.log_print('{0} │ {1} │ {2} │ {3} │ {4} │ {5} │ {6}'.format(time_str, trace_str, train_loss_str, train_loss_best_str, valid_loss_str, time_improvement_str, traces_per_sec_str))

                if save_new_artifact:
                    save_artifact(artifact, artifact_file, opt)

            util.log_print('Stopped after {0} traces'.format(trace))
    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
