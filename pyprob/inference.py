import pyprob
from pyprob import util
from pyprob.logger import Logger
from pyprob.comm import BatchRequester
from pyprob.nn import Artifact, Batch
import torch.optim as optim
from termcolor import colored
import datetime
import time
import sys
import traceback

class InferenceRemote(object):
    def __init__(self, server='tcp://127.0.0.1:5555', batch_pool=False, resume_artifact_file=None, standardize_observes=False):
        self._server = server
        self._batch_pool = batch_pool
        self._standardize_observes = standardize_observes
        self._resume_artifact_file = resume_artifact_file
        if resume_artifact_file is None:
            self._artifact = None
        else:
            self._artifact = util.load_artifact(resume_artifact_file, util.cuda_enabled, util.cuda_device)


    def compile(self, lstm_dim=512, lstm_depth=2, obs_emb='fc', obs_reshape=None, obs_emb_dim=512, smp_emb_dim=32, one_hot_dim=64, softmax_boost=20, dropout=0.2, batch_size=64, valid_size=256, valid_interval=1000, optimizer_method='adam', learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, parallelize=False, truncate_backprop=-1, grad_clip=-1, max_traces=-1, directory='.', keep_all_artifacts=False):
        time_stamp = util.get_time_stamp()
        util.logger = Logger('{0}/{1}'.format('.', 'pyprob-compile-log' + time_stamp))
        file_name = '{0}/{1}'.format(directory, 'pyprob-artifact' + time_stamp)
        with BatchRequester(self._server, self._standardize_observes, self._batch_pool) as requester:
            if self._artifact is not None:
                prev_artifact_total_traces = self._artifact.total_traces
                prev_artifact_total_iterations = self._artifact.total_iterations
                prev_artifact_total_training_seconds = self._artifact.total_training_seconds
            else:
                prev_artifact_total_traces = 0
                prev_artifact_total_iterations = 0
                prev_artifact_total_training_seconds = 0
                self._artifact = Artifact(dropout, util.cuda_enabled, util.cuda_device, self._standardize_observes, softmax_boost)
                self._artifact.set_one_hot_dims(one_hot_dim)
                traces, _ = requester.get_traces(valid_size, discard_source=True)
                self._artifact.set_valid_batch(Batch(traces))
                example_observes = self._artifact.valid_batch[0].observes_tensor
                if obs_reshape is not None:
                    try:
                        obs_reshape = [int(x) for x in obs_reshape.split('x')]
                        reshape_test = example_observes.view(obs_reshape)
                    except:
                        util.logger.log_error('Invalid obsReshape argument. Expecting a format where dimensions are separated by "x" (example: "1x10x10"). The total number of elements in the original 1d input and the requested shape should be the same (example: 100 -> "1x10x10" or "2x50").')
                self._artifact.set_observe_embedding(example_observes, obs_emb, obs_emb_dim, obs_reshape)
                self._artifact.set_sample_embedding(smp_emb_dim)
                self._artifact.set_lstm(lstm_dim, lstm_depth)
                self._artifact.polymorph()

                traces, _ = requester.get_traces(batch_size)
                batch = Batch(traces)
                loss = self._artifact.loss(batch, optimizer=None, truncate=truncate_backprop, grad_clip=grad_clip, data_parallel=parallelize)
                train_loss = loss.data[0]
                self._artifact.train_history_trace.append(0)
                self._artifact.train_history_loss.append(train_loss)
                self._artifact.train_loss_best = train_loss
                self._artifact.train_loss_worst = train_loss

                sys.stdout.write('Computing validation loss...                             \r')
                sys.stdout.flush()
                self._artifact.eval()
                valid_loss = self._artifact.valid_loss(parallelize)
                self._artifact.valid_history_trace.append(0)
                self._artifact.valid_history_loss.append(valid_loss)
                self._artifact.valid_loss_best = valid_loss
                self._artifact.valid_loss_worst = valid_loss


            train_loss_best_str = '{:+.6e}'.format(self._artifact.train_loss_best)
            train_loss_start_str = '{:+.6e}'.format(self._artifact.train_history_loss[0])
            train_loss_session_start_str = '{:+.6e}'.format(self._artifact.train_history_loss[-1])

            valid_loss_best_str = '{:+.6e}'.format(self._artifact.valid_loss_best)
            valid_loss_start_str = '{:+.6e}'.format(self._artifact.valid_history_loss[0])
            valid_loss_session_start_str = '{:+.6e}'.format(self._artifact.valid_history_loss[-1])

            if optimizer_method == 'adam':
                optimizer = optim.Adam(self._artifact.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(self._artifact.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_Decay)

            iteration = 0
            iteration_batch = 0
            trace = 0
            time_start = time.time()
            time_improvement = time_start
            time_last_batch = time_start
            time_spent_validation = -1

            # train_loss_str = '               '
            # valid_loss_str = '{:+.6e}  '.format(self._artifact.valid_history_loss[-1])
            valid_loss_str = '               '

            last_validation_trace = 0

            time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - time_start))
            time_best_str = time_str
            time_session_start_str = time_str
            time_improvement_str = util.days_hours_mins_secs(time.time() - time_improvement)
            trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))
            trace_best_str = trace_str
            trace_session_start_str = trace_str
            traces_per_sec_str = '   '

            try:
                stop = False
                while not stop:
                    save_new_artifact = False
                    iteration_batch += 1
                    traces, time_wait = requester.get_traces(batch_size)
                    batch = Batch(traces)
                    self._artifact.polymorph(batch)

                    # Time statistics
                    time_spent_last_batch = max(util.epsilon, time.time() - time_last_batch)
                    if time_spent_validation != -1:
                        time_spent_last_batch -= time_spent_validation
                    time_last_batch = time.time()
                    traces_per_sec = batch_size / time_spent_last_batch


                    iteration += 1
                    sys.stdout.write('Training...                                              \r')
                    sys.stdout.flush()

                    self._artifact.train()
                    loss = self._artifact.loss(batch, optimizer=optimizer, truncate=truncate_backprop, grad_clip=grad_clip, data_parallel=parallelize)
                    train_loss = loss.data[0]

                    trace += batch.length
                    if max_traces != -1:
                        if trace >= max_traces:
                            stop = True

                    self._artifact.total_training_seconds = prev_artifact_total_training_seconds + (time.time() - time_start)
                    self._artifact.total_iterations = prev_artifact_total_iterations + iteration
                    self._artifact.total_traces = prev_artifact_total_traces + trace

                    self._artifact.train_history_trace.append(self._artifact.total_traces)
                    self._artifact.train_history_loss.append(train_loss)

                    time_str = util.days_hours_mins_secs(prev_artifact_total_training_seconds + (time.time() - time_start))
                    trace_str = '{:5}'.format('{:,}'.format(prev_artifact_total_traces + trace))

                    # Compute validation loss as needed
                    time_spent_validation = -1
                    if (trace - last_validation_trace > valid_interval) or stop:
                        time_validation_start = time.time()

                        save_new_artifact = True
                        time_best_str = time_str
                        trace_best_str = trace_str

                        sys.stdout.write('Computing validation loss...                             \r')
                        sys.stdout.flush()

                        self._artifact.eval()
                        valid_loss = self._artifact.valid_loss(parallelize)

                        self._artifact.valid_history_trace.append(self._artifact.total_traces)
                        self._artifact.valid_history_loss.append(valid_loss)

                        if valid_loss < self._artifact.valid_loss_best:
                            self._artifact.valid_loss_best = valid_loss
                            valid_loss_str = colored('{:+.6e} ▼'.format(valid_loss), 'green', attrs=['bold'])
                            valid_loss_best_str = colored('{:+.6e}'.format(valid_loss), 'green', attrs=['bold'])
                        elif valid_loss > self._artifact.valid_loss_worst:
                            self._artifact.valid_loss_worst = valid_loss
                            valid_loss_str = colored('{:+.6e} ▲'.format(valid_loss), 'red', attrs=['bold'])
                        elif valid_loss < self._artifact.valid_history_loss[-1]:
                            valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'green')
                        elif valid_loss > self._artifact.valid_history_loss[-1]:
                            valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'red')
                        else:
                            valid_loss_str = '{:+.6e}  '.format(valid_loss)

                        last_validation_trace = trace - 1
                        time_spent_validation = time.time() - time_validation_start


                    if train_loss < self._artifact.train_loss_best:
                        self._artifact.train_loss_best = train_loss
                        train_loss_str = colored('{:+.6e} ▼'.format(train_loss), 'green', attrs=['bold'])
                        train_loss_best_str = colored('{:+.6e}'.format(self._artifact.train_loss_best), 'green', attrs=['bold'])

                        save_new_artifact = True
                        time_best_str = time_str
                        trace_best_str = trace_str

                        time_improvement = time.time()
                    elif train_loss > self._artifact.train_loss_worst:
                        self._artifact.train_loss_worst = train_loss
                        train_loss_str = colored('{:+.6e} ▲'.format(train_loss), 'red', attrs=['bold'])
                    elif train_loss < self._artifact.valid_history_loss[-1]:
                        train_loss_str = colored('{:+.6e}  '.format(train_loss), 'green')
                    elif train_loss > self._artifact.valid_history_loss[-1]:
                        train_loss_str = colored('{:+.6e}  '.format(train_loss), 'red')
                    else:
                        train_loss_str = '{:+.6e}  '.format(train_loss)


                    traces_per_sec_str = '{:3}'.format('{:,}'.format(int(traces_per_sec)))

                    time_improvement_str = util.days_hours_mins_secs(time.time() - time_improvement)
                    util.logger.reset()
                    util.logger.log('────────┬─' + '─'*len(time_str) + '─┬─' + '─'*len(trace_str) + '─┬─────────────────┬─────────────────')
                    util.logger.log('        │ {0:>{1}} │ {2:>{3}} │ Training loss   │ Valid. loss     '.format('Train. time', len(time_str), 'Trace', len(trace_str)))
                    util.logger.log('────────┼─' + '─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────')
                    util.logger.log('Start   │ {0:>{1}} │ {2:>{3}} │ {4}   │ {5}'.format(time_session_start_str, len(time_str), trace_session_start_str, len(trace_str), train_loss_session_start_str, valid_loss_session_start_str))
                    util.logger.log('Best    │ {0:>{1}} │ {2:>{3}} │ {4}   │ {5}'.format(time_best_str, len(time_str), trace_best_str, len(trace_str), train_loss_best_str, valid_loss_best_str))
                    util.logger.log('Current │ {0} │ {1} │ {2} │ {3}'.format(time_str, trace_str, train_loss_str, valid_loss_str))
                    util.logger.log('────────┴─' + '─'*len(time_str) + '─┴─' + '─'*len(trace_str) + '─┴─────────────────┴─────────────────')
                    util.logger.log('Training on {0}, {1} traces/s'.format('CUDA' if util.cuda_enabled else 'CPU', traces_per_sec_str))
                    util.logger.update()

                    if save_new_artifact:
                        self._artifact.optimizer = optimizer_method
                        if keep_all_artifacts:
                            time_stamp = util.get_time_stamp()
                            file_name = '{0}/{1}'.format(directory, 'pyprob-artifact' + time_stamp)
                        util.save_artifact(self._artifact, file_name)


            except KeyboardInterrupt:
                util.logger.log('Stopped')
                util.logger.update()
            except Exception:
                traceback.print_exc(file=sys.stdout)
