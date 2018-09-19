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

    def prior_distribution(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=lambda trace: trace.result, *args, **kwargs):
        return self.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, *args, **kwargs)

    def posterior_traces(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=None, observe=None, *args, **kwargs):
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
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

            log_weights = None
            name = 'Posterior, {} Metropolis Hastings, num_traces={:,}, accepted={:,.2f}%, sample_reuse={:,.2f}%'.format('lightweight' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'random-walk', num_traces, 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all)

        return Empirical(traces, log_weights, name=name)

    def posterior_distribution(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=lambda trace: trace.result, observe=None, *args, **kwargs):
        return self.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=map_func, observe=observe, *args, **kwargs)

    def learn_inference_network(self, num_traces=None, inference_network=InferenceNetwork.FEEDFORWARD, prior_inflation=PriorInflation.DISABLED, trace_store_dir=None, observe_embeddings={}, batch_size=64, valid_batch_size=64, valid_interval=1000, learning_rate=0.0001, weight_decay=1e-5):
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
        self._inference_network.optimize(num_traces, batch_generator, batch_size=batch_size, valid_interval=valid_interval, learning_rate=learning_rate, weight_decay=weight_decay)

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
        self._model_server = ModelServer(server_address)
        super().__init__('{} running on {}'.format(self._model_server.model_name, self._model_server.system_name))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self._model_server.close()

    def forward(self):
        return self._model_server.forward()
