import torch
import torch.nn as nn
import time
import sys
import os
import uuid
from threading import Thread
from termcolor import colored
import math
import random
import tarfile
import tempfile
import shutil

from .distributions import Empirical
from . import state, util, __version__, TraceMode, InferenceEngine, InferenceNetwork, PriorInflation, Optimizer, TrainingObservation
from .nn import ObserveEmbedding, SampleEmbedding, Batch, InferenceNetworkSimple, InferenceNetworkLSTM
from .remote import ModelServer
from .analytics import save_report


class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._inference_network = None
        self._trace_cache_path = None
        self._trace_cache = []
        self._trace_cache_discarded_file_names = []

    def forward(self):
        raise NotImplementedError()

    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, metropolis_hastings_trace=None, observation_importance_exponent=1., *args, **kwargs):
        while True:
            if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                self._inference_network.new_trace(util.pack_observes_to_variable(kwargs['observation']).unsqueeze(0))
            state.begin_trace(self.forward, trace_mode, prior_inflation, inference_engine, inference_network, metropolis_hastings_trace, observation_importance_exponent)
            result = self.forward(*args, **kwargs)
            trace = state.end_trace(result)
            yield trace

    def _trace_result_generator(self, prior_inflation=PriorInflation.DISABLED, observation_importance_exponent=1., *args, **kwargs):
        while True:
            state.begin_trace(None, trace_mode=TraceMode.NONE, prior_inflation=prior_inflation, observation_importance_exponent=observation_importance_exponent)
            result = self.forward(*args, **kwargs)
            state.end_trace(None)
            yield result

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, observation_importance_exponent=1., *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observation_importance_exponent=observation_importance_exponent, *args, **kwargs)
        ret = []
        time_start = time.time()
        if ((trace_mode != TraceMode.PRIOR) and (util.verbosity > 1)) or (util.verbosity > 2):
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
            prev_duration = 0
        for i in range(num_traces):
            if ((trace_mode != TraceMode.PRIOR) and (util.verbosity > 1)) or (util.verbosity > 2):
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()
            trace = next(generator)
            if map_func is not None:
                ret.append(map_func(trace))
            else:
                ret.append(trace)
        if ((trace_mode != TraceMode.PRIOR) and (util.verbosity > 1)) or (util.verbosity > 2):
            print()
        return ret

    def prior_sample(self, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        generator = self._trace_result_generator(prior_inflation, *args, **kwargs)
        return next(generator)

    def prior_distribution(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        generator = self._trace_result_generator(prior_inflation, *args, **kwargs)
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
        return Empirical(ret, name='Prior, num_traces={:,}'.format(num_traces))

    def posterior_distribution(self, *args, **kwargs):
        return self.posterior_traces(*args, **kwargs).map(lambda x: x.result)

    def posterior_traces(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, burn_in=None, initial_trace=None, observation_importance_exponent=1., *args, **kwargs):
        if (inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK) and (self._inference_network is None):
            raise RuntimeError('Cannot run inference with inference network because there is none available. Use learn_inference_network first.')
        if burn_in is not None:
            if burn_in >= num_traces:
                raise ValueError('burn_in must be less than num_traces')
        else:
            # Default burn_in
            burn_in = int(min(num_traces / 10, 1000))

        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
            traces = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, observation_importance_exponent=observation_importance_exponent, *args, **kwargs)
            log_weights = [trace.log_importance_weight for trace in traces]
            name = 'Posterior, importance sampling (with proposal = prior), num_traces={:,}'.format(num_traces)
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            self._inference_network.eval()
            traces = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, observation_importance_exponent=observation_importance_exponent, *args, **kwargs)
            log_weights = [trace.log_importance_weight for trace in traces]
            name = 'Posterior, importance sampling (with learned proposal, training_traces={:,}), num_traces={:,}'.format(self._inference_network._total_train_traces, num_traces)
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
            traces = []
            if initial_trace is None:
                current_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observation_importance_exponent=observation_importance_exponent, *args, **kwargs))
            else:
                current_trace = initial_trace

            time_start = time.time()
            traces_accepted = 0
            samples_reused = 0
            samples_all = 0
            if util.verbosity > 1:
                len_str_num_traces = len(str(num_traces))
                print('Time spent  | Time remain.| Progress             | {} | Accepted|Smp reuse| Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
                prev_duration = 0
            for i in range(num_traces):
                if util.verbosity > 1:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        print('{} | {} | {} | {}/{} | {} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:,.2f}%'.format(100 * (traces_accepted / (i + 1))).rjust(7), '{:,.2f}%'.format(100 * samples_reused / max(1, samples_all)).rjust(7), traces_per_second), end='\r')
                        sys.stdout.flush()
                candidate_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, metropolis_hastings_trace=current_trace, observation_importance_exponent=observation_importance_exponent, *args, **kwargs))
                log_acceptance_ratio = math.log(current_trace.length) - math.log(candidate_trace.length) + candidate_trace.log_prob_observed - current_trace.log_prob_observed
                for sample in candidate_trace.samples:
                    if sample.reused:
                        log_acceptance_ratio += util.safe_torch_sum(sample.log_prob)
                        log_acceptance_ratio -= util.safe_torch_sum(current_trace._samples_all_dict_address[sample.address].log_prob)
                        samples_reused += 1
                samples_all += candidate_trace.length

                if state._metropolis_hastings_site_transition_log_prob is None:
                    print(colored('Warning: trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls', 'red', attrs=['bold']))
                else:
                    log_acceptance_ratio += util.safe_torch_sum(state._metropolis_hastings_site_transition_log_prob)

                # print(log_acceptance_ratio)
                if math.log(random.random()) < float(log_acceptance_ratio):
                    traces_accepted += 1
                    current_trace = candidate_trace
                traces.append(current_trace)
            if util.verbosity > 1:
                print()
            if burn_in is not None:
                traces = traces[burn_in:]
            log_weights = None
            name = 'Posterior, {} Metropolis Hastings, num_traces={:,}, burn_in={:,}, accepted={:,.2f}%, sample_reuse={:,.2f}%'.format('lightweight' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'random-walk', num_traces, burn_in, 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all)

        # if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            # ret.name += ' (effective sample size: {:,.2f})'.format(float(ret.effective_sample_size))
        return Empirical(traces, log_weights, name=name)

    def learn_inference_network(self, inference_network=InferenceNetwork.LSTM, training_observation=TrainingObservation.OBSERVE_DIST_SAMPLE, prior_inflation=PriorInflation.DISABLED, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_reshape=None, observe_embedding_dim=128, sample_embedding=SampleEmbedding.FULLY_CONNECTED, lstm_dim=128, lstm_depth=2, sample_embedding_dim=16, address_embedding_dim=128, batch_size=64, valid_size=256, valid_interval=2048, optimizer_type=Optimizer.ADAM, learning_rate=0.0001, momentum=0.9, weight_decay=1e-5, num_traces=-1, use_trace_cache=False, auto_save=False, auto_save_file_name='pyprob_inference_network', *args, **kwargs):
        if use_trace_cache and self._trace_cache_path is None:
            print('Warning: There is no trace cache assigned, training with online trace generation.')
            use_trace_cache = False

        if use_trace_cache:
            print('Using trace cache to train...')

            def new_batch_func(size=batch_size, discard_source=False):
                if discard_source:
                    self._trace_cache = []

                while len(self._trace_cache) < size:
                    current_files = self._trace_cache_current_files()
                    if len(current_files) == 0:
                        cache_is_empty = True
                        cache_was_empty = False
                        while cache_is_empty:
                            current_files = self._trace_cache_current_files()
                            num_files = len(current_files)
                            if num_files > 0:
                                cache_is_empty = False
                                if cache_was_empty:
                                    print('Resuming, new data appeared in trace cache (currently with {} files) at {}'.format(num_files, self._trace_cache_path))
                            else:
                                if not cache_was_empty:
                                    print('Waiting for new data, empty (or fully discarded) trace cache at {}'.format(self._trace_cache_path))
                                    cache_was_empty = True
                                time.sleep(0.5)

                    current_file = random.choice(current_files)
                    if discard_source:
                        self._trace_cache_discarded_file_names.append(current_file)
                    new_traces = self._load_traces(current_file)
                    if len(new_traces) == 0:  # When empty or corrupt file is read
                        self._trace_cache_discarded_file_names.append(current_file)
                    else:
                        random.shuffle(new_traces)
                        self._trace_cache += new_traces

                traces = self._trace_cache[0:size]
                self._trace_cache[0:size] = []
                return Batch(traces)
        else:
            def new_batch_func(size=batch_size, discard_source=False):
                traces = self._traces(size, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, *args, **kwargs)
                return Batch(traces)

        if self._inference_network is None:
            print('Creating new inference network...')
            valid_batch = new_batch_func(valid_size, discard_source=True)
            if inference_network == InferenceNetwork.SIMPLE:
                self._inference_network = InferenceNetworkSimple(model_name=self.name, observe_embedding=observe_embedding, observe_reshape=observe_reshape, observe_embedding_dim=observe_embedding_dim, valid_batch=valid_batch)
            else:  # inference_network == InferenceNetwork.LSTM:
                self._inference_network = InferenceNetworkLSTM(model_name=self.name, lstm_dim=lstm_dim, lstm_depth=lstm_depth, observe_embedding=observe_embedding, observe_reshape=observe_reshape, observe_embedding_dim=observe_embedding_dim, sample_embedding=sample_embedding, sample_embedding_dim=sample_embedding_dim, address_embedding_dim=address_embedding_dim, valid_batch=valid_batch)

            self._inference_network.polymorph()
        else:
            print('Continuing to train existing inference network...')

        self._inference_network.train()
        self._inference_network.optimize(new_batch_func, training_observation, optimizer_type, num_traces, learning_rate, momentum, weight_decay, valid_interval, auto_save, auto_save_file_name)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network._save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetworkSimple._load(file_name, util._cuda_enabled, util._cuda_device)

    def trace_length_mean(self, num_traces=1000, *args, **kwargs):
        trace_lengths = self._traces(num_traces, trace_mode=TraceMode.PRIOR, map_func=lambda trace: trace.length, *args, **kwargs)
        trace_length_dist = Empirical(trace_lengths)
        return trace_length_dist.mean

    def trace_length_stddev(self, num_traces=1000, *args, **kwargs):
        trace_lengths = self._traces(num_traces, trace_mode=TraceMode.PRIOR, map_func=lambda trace: trace.length, *args, **kwargs)
        trace_length_dist = Empirical(trace_lengths)
        return trace_length_dist.stddev

    def trace_length_min(self, num_traces=1000, *args, **kwargs):
        trace_lengths = self._traces(num_traces, trace_mode=TraceMode.PRIOR, map_func=lambda trace: trace.length, *args, **kwargs)
        trace_length_dist = Empirical(trace_lengths)
        return trace_length_dist.min

    def trace_length_max(self, num_traces=1000, *args, **kwargs):
        trace_lengths = self._traces(num_traces, trace_mode=TraceMode.PRIOR, map_func=lambda trace: trace.length, *args, **kwargs)
        trace_length_dist = Empirical(trace_lengths)
        return trace_length_dist.max

    def save_trace_cache(self, trace_cache_path, files=16, traces_per_file=512, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        f = 0
        done = False
        while not done:
            traces = self._traces(traces_per_file, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, *args, **kwargs)
            file_name = os.path.join(trace_cache_path, 'pyprob_traces_{}_{}'.format(traces_per_file, str(uuid.uuid4())))
            self._save_traces(traces, file_name)
            f += 1
            if (files != -1) and (f >= files):
                done = True

    def use_trace_cache(self, trace_cache_path):
        self._trace_cache_path = trace_cache_path
        num_files = len(self._trace_cache_current_files())
        print('Monitoring trace cache (currently with {} files) at {}'.format(num_files, trace_cache_path))

    def _trace_cache_current_files(self):
        files = [name for name in os.listdir(self._trace_cache_path)]
        files = list(map(lambda f: os.path.join(self._trace_cache_path, f), files))
        for discarded_file_name in self._trace_cache_discarded_file_names:
            if discarded_file_name in files:
                files.remove(discarded_file_name)
        return files

    def _save_traces(self, traces, file_name):
        data = {}
        data['traces'] = traces
        data['length'] = len(traces)
        data['model_name'] = self.name
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_traces')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_traces')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    def _load_traces(self, file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_traces')
            tar.extract('pyprob_traces', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
            shutil.rmtree(tmp_dir)
        except:
            print('Warning: cannot load traces from file, file potentially corrupt: {}'.format(file_name))
            return []

        # print('Loading trace cache of length {}'.format(data['length']))
        if data['model_name'] != self.name:
            print(colored('Warning: different model names (loaded traces: {}, current model: {})'.format(data['model_name'], self.name), 'red', attrs=['bold']))
        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded traces: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded traces: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        traces = data['traces']
        if util._cuda_enabled:
            for trace in traces:
                trace.cuda()
        return data['traces']

    def save_analytics(self, file_name, detailed_traces=2):
        if self._inference_network is None:
            raise RuntimeError('Analytics is currently available only with a trained inference network. Use learn_inference_network first.')
        save_report(self, file_name, detailed_traces=detailed_traces)


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

    def forward(self, observation=None):
        return self._model_server.forward(observation)
