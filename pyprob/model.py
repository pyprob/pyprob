#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
from pyprob.logger import Logger
from pyprob.comm import BatchRequester, ProposalReplier
from pyprob.nn import Artifact, Batch
from pyprob.state import TraceMode
import torch.optim as optim
from torch.autograd import Variable
from termcolor import colored
import numpy as np
from scipy.misc import logsumexp
import datetime
import time
import sys
import traceback

class Model(object):
    def __init__(self, model_func, default_observes=[], standardize_observes=False, directory='.', resume=False):
        self._model_func = model_func
        self._default_observes = default_observes
        self._standardize_observes = standardize_observes
        self._file_name = '{0}/{1}'.format(directory, 'pyprob-artifact' + util.get_time_stamp())

        util.logger.reset()
        util.logger.log_config()

        if resume:
            resume_artifact_file = util.file_starting_with('{0}/{1}'.format(directory, 'pyprob-artifact'), -1)
            util.logger.log(colored('Resuming previous artifact: {}'.format(resume_artifact_file), 'blue', attrs=['bold']))
            self._artifact = util.load_artifact(resume_artifact_file, util.cuda_enabled, util.cuda_device)
        else:
            self._artifact = None

    def learn_proposal(self, lstm_dim=512, lstm_depth=2, obs_emb='fc', obs_reshape=None, obs_emb_dim=512, smp_emb_dim=32, one_hot_dim=64, softmax_boost=20, mixture_components=10, dropout=0.2,batch_size=64, valid_interval=1000, optimizer_method='adam', learning_rate=0.0001, momentum=0.9, weight_decay=0.0005, parallelize=False, truncate_backprop=-1, grad_clip=-1, max_traces=-1, keep_all_artifacts=False, replace_valid_batch=False, valid_size=256):

        if self._artifact is None:
            util.logger.log(colored('Creating new artifact...', 'blue', attrs=['bold']))
            self._artifact = Artifact(dropout, util.cuda_enabled, util.cuda_device, self._standardize_observes, softmax_boost, mixture_components)
            self._artifact.set_one_hot_dims(one_hot_dim)

            #pyprob.state.set_mode('compilation')
            pyprob.state.set_mode(TraceMode.COMPILATION)
            traces = self.prior_traces(valid_size, self._default_observes)
            #pyprob.state.set_mode('inference')
            pyprob.state.set_mode(TraceMode.INFERENCE)

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

            #pyprob.state.set_mode('compilation')
            pyprob.state.set_mode(TraceMode.COMPILATION)
            traces = self.prior_traces(valid_size, self._default_observes)
            #pyprob.state.set_mode('inference')
            pyprob.state.set_mode(TraceMode.INFERENCE)

            batch = Batch(traces)
            self._artifact.polymorph(batch)
            loss = self._artifact.loss(batch, optimizer=None)
            train_loss = loss.data[0]
            self._artifact.train_history_trace.append(0)
            self._artifact.train_history_loss.append(train_loss)
            self._artifact.train_loss_best = train_loss
            self._artifact.train_loss_worst = train_loss

            sys.stdout.write('Computing validation loss...                             \r')
            sys.stdout.flush()
            self._artifact.eval()
            valid_loss = self._artifact.valid_loss()
            sys.stdout.write('                                                         \r')
            sys.stdout.flush()
            self._artifact.valid_history_trace.append(0)
            self._artifact.valid_history_loss.append(valid_loss)
            self._artifact.valid_loss_best = valid_loss
            self._artifact.valid_loss_worst = valid_loss

        # Compilation
        util.logger.log(colored('New artifact will be saved to: {}'.format(self._file_name), 'blue', attrs=['bold']))
        if replace_valid_batch:
            util.logger.log(colored('Replacing the validation batch of the artifact', 'magenta', attrs=['bold']))
            self._artifact.valid_size = valid_size
            #pyprob.state.set_mode('compilation')
            pyprob.state.set_mode(TraceMode.COMPILATION)
            traces = self.prior_traces(valid_size, self._default_observes)
            #pyprob.state.set_mode('inference')
            pyprob.state.set_mode(TraceMode.INFERENCE)
            self._artifact.set_valid_batch(Batch(traces))

        prev_artifact_total_traces = self._artifact.total_traces
        prev_artifact_total_iterations = self._artifact.total_iterations
        prev_artifact_total_training_seconds = self._artifact.total_training_seconds

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
        # valid_loss_str = '               '
        valid_loss_str = '{:+.6e}  '.format(self._artifact.valid_history_loss[-1])

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
            util.logger.log_compile_begin('pyprob model', time_str, time_improvement_str, trace_str, traces_per_sec_str)
            stop = False
            while not stop:
                save_new_artifact = False
                iteration_batch += 1
                #pyprob.state.set_mode('compilation')
                pyprob.state.set_mode(TraceMode.COMPILATION)
                traces = self.prior_traces(valid_size, self._default_observes)
                #pyprob.state.set_mode('inference')
                pyprob.state.set_mode(TraceMode.INFERENCE)
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
                    util.logger.log_compile_valid(time_str, time_improvement_str, trace_str, traces_per_sec_str)

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

                util.logger.log_compile(time_str, time_session_start_str, time_best_str, time_improvement_str, trace_str, trace_session_start_str, trace_best_str, train_loss_str, train_loss_session_start_str, train_loss_best_str, valid_loss_str, valid_loss_session_start_str, valid_loss_best_str, traces_per_sec_str)

                if save_new_artifact:
                    self._artifact.optimizer = optimizer_method
                    if keep_all_artifacts:
                        self._file_name = '{0}/{1}'.format(directory, 'pyprob-artifact' + util.get_time_stamp())
                    util.save_artifact(self._artifact, self._file_name)

        except KeyboardInterrupt:
            util.logger.log('Stopped')
            util.logger._jupyter_update()
        except Exception:
            traceback.print_exc(file=sys.stdout)

    def prior_sample(self, *args, **kwargs):
        while True:
            yield self._model_func(*args, **kwargs)
    def prior_samples(self, samples=1, *args, **kwargs):
        generator = self.prior_sample(*args, **kwargs)
        return [next(generator) for i in range(samples)]
    def prior_trace_guided(self, *args, **kwargs):
        pyprob.state.set_artifact(self._artifact)
        while True:
            self._artifact.new_trace(Variable(util.pack_observes_to_tensor(args[0]).unsqueeze(0), volatile=True))
            #pyprob.state.set_mode('compiled_inference')
            pyprob.state.set_mode(TraceMode.COMPILED_INFERENCE)
            pyprob.state.begin_trace(self._model_func)
            res = self._model_func(*args, **kwargs)
            trace = pyprob.state.end_trace()
            #pyprob.state.set_mode('inference')
            pyprob.state.set_mode(TraceMode.INFERENCE)
            trace.set_result(res)
            yield trace
    def prior_traces_guided(self, samples=1, *args, **kwargs):
        generator = self.prior_trace_guided(*args, **kwargs)
        return [next(generator) for i in range(samples)]
    def prior_trace(self, *args, **kwargs):
        while True:
            pyprob.state.begin_trace(self._model_func)
            res = self._model_func(*args, **kwargs)
            trace = pyprob.state.end_trace()
            trace.set_result(res)
            yield trace
    def prior_traces(self, samples=1, *args, **kwargs):
        generator = self.prior_trace(*args, **kwargs)
        return [next(generator) for i in range(samples)]
    def posterior_samples(self, samples=10, *args, **kwargs):
        if self._artifact is None:
            traces = self.prior_traces(samples, *args, **kwargs)
            weights = np.array([trace.log_p for trace in traces])
        else:
            traces = self.prior_traces_guided(samples, *args, **kwargs)
            weights = np.array([trace.log_p for trace in traces])
        results = [trace.result for trace in traces]
        return pyprob.distributions.Empirical(results, weights)



class RemoteModel(Model):
    def __init__(self, local_server='tcp://0.0.0.0:6666', remote_server='tcp://127.0.0.1:5555', batch_pool=False):
        self._local_server = local_server
        self._remote_server = remote_server
        self._batch_pool = batch_pool

    def prior_traces(self, samples=1, *args, **kwargs):
        with BatchRequester(self._remote_server, self._standardize_observes, self._batch_pool) as requester:
            traces, _ = requester.get_traces(samples, discard_source=False)
            return traces

    def infer(self):
        with ProposalReplier(self._local_server) as replier:
            util.logger.log(self._artifact.get_info())
            util.logger.log()
            util.logger.log(colored('Inference engine running at ' + self._local_server, 'blue', attrs=['bold']))
            self._artifact.eval()

            time_last_new_trace = time.time()
            duration_last_trace = 0
            traces_per_sec = 0
            max_traces_per_sec = 0
            traces_per_sec_str = '-       '
            max_traces_per_sec_str = '-       '

            try:
                total_traces = 0
                observes = None
                util.logger.log_infer_begin()
                while True:
                    util.logger.log_infer(traces_per_sec_str, max_traces_per_sec_str, total_traces)
                    replier.receive_request(self._artifact.standardize_observes)
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
                        observes = Variable(replier.observes.unsqueeze(0), volatile=True)
                        replier.reply_observes_received()
                        self._artifact.new_trace(observes)
                    else:
                        proposal_distribution = self._artifact.forward(replier.previous_sample, replier.current_sample, volatile=True)
                        replier.reply_proposal(proposal_distribution)

            except KeyboardInterrupt:
                util.logger.log('Stopped')
                util.logger._jupyter_update()
            except Exception:
                traceback.print_exc(file=sys.stdout)
