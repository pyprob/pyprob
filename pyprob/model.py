import torch
import time
import sys
import os
import math
import random
from termcolor import colored

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine, InferenceNetwork
from .nn import BatchGenerator, InferenceNetworkFeedForward
from .remote import ModelServer


class Model():
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._inference_network = None

    def forward(self):
        raise NotImplementedError()

    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, *args, **kwargs):
        while True:
            state.begin_trace(self.forward, trace_mode, prior_inflation, inference_engine, inference_network, observe, metropolis_hastings_trace)
            result = self.forward(*args, **kwargs)
            trace = state.end_trace(result)
            yield trace

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, silent=False, observe=None, *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, *args, **kwargs)
        traces = []
        log_weights = []
        time_start = time.time()
        if (util._verbosity > 1) and not silent:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
            prev_duration = 0
        for i in range(num_traces):
            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()
            trace = next(generator)
            if map_func is not None:
                traces.append(map_func(trace))
            else:
                traces.append(trace)
            log_weights.append(trace.log_importance_weight)
        if (util._verbosity > 1) and not silent:
            print()
        return traces, log_weights

    def prior_traces(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=None, *args, **kwargs):
        traces, _ = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, map_func=map_func, *args, **kwargs)
        return Empirical(traces, name='Prior, num_traces={:,}'.format(num_traces))

<<<<<<< HEAD
    def prior_distribution(self, num_traces=1000, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        ret = []
        time_start = time.time()
        if util.verbosity > 1:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
        prev_duration = 0
        for i in range(num_traces):
            if util.verbosity > 1:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()
            ret.append(next(generator))
        if util.verbosity > 1:
            print()
        return Empirical(ret, name='Prior, num_traces={}'.format(num_traces))

    def posterior_distribution(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, burn_in=None, *args, **kwargs):
        if (inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK) and (self._inference_network is None):
            raise RuntimeError('Cannot run inference with inference network because there is none available. Use learn_inference_network first.')
        if burn_in is not None:
            if burn_in >= num_traces:
                raise ValueError('burn_in must be less than num_traces')
        else:
            # Default burn_in
            burn_in = int(min(num_traces / 10, 1000))
=======
    def prior_distribution(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=lambda trace: trace.result, *args, **kwargs):
        return self.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, *args, **kwargs)
>>>>>>> origin/master

    def posterior_traces(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=None, observe=None, *args, **kwargs):
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
<<<<<<< HEAD
            ret = self._prior_traces(num_traces, trace_mode=TraceMode.IMPORTANCE_SAMPLING_WITH_PRIOR, inference_network=None, map_func=lambda trace: (trace.log_importance_weight, trace.result), *args, **kwargs)
            ret = [list(t) for t in zip(*ret)]
            log_weights = ret[0]
            results = ret[1]
            name = 'Posterior, importance sampling (with prior), num_traces={}'.format(num_traces)
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            self._inference_network.eval()
            ret = self._prior_traces(num_traces, trace_mode=TraceMode.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, inference_network=self._inference_network, map_func=lambda trace: (trace.log_importance_weight, trace.result), *args, **kwargs)
            ret = [list(t) for t in zip(*ret)]
            log_weights = ret[0]
            results = ret[1]
            name = 'Posterior, importance sampling (with learned proposal), num_traces={}'.format(num_traces)
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS
            results = []
            current_trace = next(self._prior_trace_generator(trace_mode=TraceMode.LIGHTWEIGHT_METROPOLIS_HASTINGS, *args, **kwargs))
=======
            traces, log_weights = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, map_func=map_func, observe=observe, *args, **kwargs)
            name = 'Posterior, importance sampling (prior as proposal, num_traces: {:,})'.format(num_traces)
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            if self._inference_network is None:
                raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
            traces, log_weights = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=observe, *args, **kwargs)
            name = 'Posterior, importance sampling with inference network (learned proposal, num_traces: {:,}, training_traces: {})'.format(num_traces, self._inference_network._total_train_traces)
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
            traces = []
            if initial_trace is None:
                current_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observe=observe, *args, **kwargs))
            else:
                current_trace = initial_trace

>>>>>>> origin/master
            time_start = time.time()
            traces_accepted = 0
            samples_reused = 0
            samples_all = 0
            if util._verbosity > 1:
                len_str_num_traces = len(str(num_traces))
                print('Time spent  | Time remain.| Progress             | {} | Accepted|Smp reuse| Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
                prev_duration = 0
            for i in range(num_traces):
                if util._verbosity > 1:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        print('{} | {} | {} | {}/{} | {} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:,.2f}%'.format(100 * (traces_accepted / (i + 1))).rjust(7), '{:,.2f}%'.format(100 * samples_reused / max(1, samples_all)).rjust(7), traces_per_second), end='\r')
                        sys.stdout.flush()
                candidate_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, metropolis_hastings_trace=current_trace, observe=observe, *args, **kwargs))
                log_acceptance_ratio = math.log(current_trace.length_controlled) - math.log(candidate_trace.length_controlled) + candidate_trace.log_prob_observed - current_trace.log_prob_observed
                for variable in candidate_trace.variables_controlled:
                    if variable.reused:
                        log_acceptance_ratio += torch.sum(variable.log_prob)
                        log_acceptance_ratio -= torch.sum(current_trace.variables_dict_address[variable.address].log_prob)
                        samples_reused += 1
                samples_all += candidate_trace.length_controlled

                if state._metropolis_hastings_site_transition_log_prob is None:
                    print(colored('Warning: trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls', 'red', attrs=['bold']))
                else:
                    log_acceptance_ratio += torch.sum(state._metropolis_hastings_site_transition_log_prob)

                # print(log_acceptance_ratio)
                if math.log(random.random()) < float(log_acceptance_ratio):
                    traces_accepted += 1
                    current_trace = candidate_trace
                if map_func is not None:
                    traces.append(map_func(current_trace))
                else:
                    traces.append(current_trace)
            if util._verbosity > 1:
                print()
<<<<<<< HEAD
            if burn_in is not None:
                results = results[burn_in:]
            log_weights = None
            name = 'Posterior, Metropolis Hastings, num_traces={}, burn_in={}, accepted={:,.2f}%, sample_reuse={:,.2f}%'.format(num_traces, burn_in, 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all)

        return Empirical(results, log_weights, name=name)

    def learn_inference_network(self, lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_reshape=None, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=256, batch_size=64, valid_size=256, valid_interval=2048, optimizer_type=Optimizer.ADAM, learning_rate=0.0001, momentum=0.9, weight_decay=1e-4, num_traces=-1, use_trace_cache=False, auto_save=False, auto_save_file_name='pyprob_inference_network', *args, **kwargs):
        if use_trace_cache and self._trace_cache_path is None:
            print('Warning: There is no trace cache assigned, training with online trace generation.')
            use_trace_cache = False

        if use_trace_cache:
            print('Using trace cache to train...')

            def new_batch_func(size=batch_size, discard_source=False):
                current_files = self._trace_cache_current_files()
                if discard_source:
                    self._trace_cache = []
                if (len(self._trace_cache) == 0) and (len(current_files) == 0):
                    cache_is_empty = True
                    cache_was_empty = False
                    while cache_is_empty:
                        num_files = len(self._trace_cache_current_files())
                        if num_files > 0:
                            cache_is_empty = False
                            if cache_was_empty:
                                print('Resuming, new data appeared in trace cache (currently with {} files) at {}'.format(num_files, self._trace_cache_path))
                        else:
                            if not cache_was_empty:
                                print('Waiting for new data, empty trace cache at {}'.format(self._trace_cache_path))
                                cache_was_empty = True
                            time.sleep(0.5)
=======

            log_weights = None
            name = 'Posterior, {} Metropolis Hastings, num_traces={:,}, accepted={:,.2f}%, sample_reuse={:,.2f}%'.format('lightweight' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'random-walk', num_traces, 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all)
>>>>>>> origin/master

        return Empirical(traces, log_weights, name=name)

    def posterior_distribution(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=lambda trace: trace.result, observe=None, *args, **kwargs):
        return self.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=map_func, observe=observe, *args, **kwargs)

    def learn_inference_network(self, num_traces=None, inference_network=InferenceNetwork.FEEDFORWARD, prior_inflation=PriorInflation.DISABLED, trace_store_dir=None, observe_embeddings={}, batch_size=64, valid_batch_size=64, valid_interval=5000, learning_rate=0.0001, weight_decay=1e-5, auto_save_file_name=None, auto_save_interval_sec=600):
        if self._inference_network is None:
            print('Creating new inference network...')
            if inference_network == InferenceNetwork.FEEDFORWARD:
                self._inference_network = InferenceNetworkFeedForward(model=self, observe_embeddings=observe_embeddings, valid_batch_size=valid_batch_size)
            else:
                raise ValueError('Unknown inference_network: {}'.format(inference_network))
        else:
            print('Continuing to train existing inference network...')
            print('Total number of parameters: {:,}'.format(self._inference_network._history_num_params[-1]))

        batch_generator = BatchGenerator(self, prior_inflation, trace_store_dir)
        self._inference_network.to(device=util._device)
        self._inference_network.optimize(num_traces, batch_generator, batch_size=batch_size, valid_interval=valid_interval, learning_rate=learning_rate, weight_decay=weight_decay, auto_save_file_name=auto_save_file_name, auto_save_interval_sec=auto_save_interval_sec)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network._save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetworkFeedForward._load(file_name)
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        self._inference_network._model = self

    def save_trace_store(self, trace_store_dir, files=16, traces_per_file=16, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        if not os.path.exists(trace_store_dir):
            print('Directory does not exist, creating: {}'.format(trace_store_dir))
            os.makedirs(trace_store_dir)
        batch_generator = BatchGenerator(self, prior_inflation)
        batch_generator.save_trace_store(trace_store_dir, files, traces_per_file)


class ModelRemote(Model):
    def __init__(self, server_address='tcp://127.0.0.1:5555'):
        self._server_address = server_address
        self._model_server = None
        super().__init__('ModelRemote')

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self._model_server is not None:
            self._model_server.close()

    def forward(self):
        if self._model_server is None:
            self._model_server = ModelServer(self._server_address)
            self.name = '{} running on {}'.format(self._model_server.model_name, self._model_server.system_name)

        return self._model_server.forward()
