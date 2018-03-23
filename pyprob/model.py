import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import uuid
from threading import Thread
from termcolor import colored
import random
import tarfile
import tempfile
import shutil
import random
from random import randint
import math
import pdb

from .distributions import Empirical
from . import state, util, __version__, TraceMode, InferenceEngine
from .nn import ObserveEmbedding, SampleEmbedding, Batch, InferenceNetwork
from .remote import ModelServer


class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._inference_network = None
        self._trace_cache_path = None
        self._trace_cache = []

    def forward(self):
        raise NotImplementedError()

    def _prior_trace_generator(self, trace_state=TraceMode.RECORD, proposal_network=None, continuation_address=None, previous_trace=None, *args, **kwargs):
        while True:
            if trace_mode == TraceMode.RECORD_USE_INFERENCE_NETWORK:
                self._inference_network.new_trace(util.pack_observes_to_variable(kwargs['observation']).unsqueeze(0))
            state.begin_trace(self.forward, trace_mode, proposal_network, continuation_address, previous_trace)
            res = self.forward(*args, **kwargs)
            trace = state.end_trace(res)
            yield trace

    def _prior_sample_generator(self, *args, **kwargs):
        while True:
            yield self.forward(*args, **kwargs)

    def _prior_traces(self, traces=10, trace_mode=TraceMode.RECORD, proposal_network=None, *args, **kwargs):
        generator = self._prior_trace_generator(trace_mode, proposal_network, *args, **kwargs)
        ret = []
        time_start = time.time()
        for i in range(traces):
            if ((trace_mode != TraceMode.RECORD_TRAIN_INFERENCE_NETWORK) and (util.verbosity > 1)) or (util.verbosity > 2):
                duration = time.time() - time_start
                print('                                                                \r{} | {} | {} / {} | {:,.2f} traces/s'.format(util.days_hours_mins_secs_str(duration), util.progress_bar(i+1, traces), i+1, traces, i / duration), end='\r')
                sys.stdout.flush()
            ret.append(next(generator))
        if ((trace_mode != TraceMode.RECORD_TRAIN_INFERENCE_NETWORK) and (util.verbosity > 1)) or (util.verbosity > 2):
            print()
        return ret

    def _single_trace(self, *args, **kwargs):
        generator = self._prior_trace_generator(*args, **kwargs)
        return next(generator)

    def _continue_trace_at_address(self, continuation_address, previous_trace, *args, **kwargs):
        #  generator = self._prior_trace_generator(*args, **kwargs)
        generator = self._prior_trace_generator(continuation_address=continuation_address, previous_trace=previous_trace, *args, **kwargs)
        return next(generator)

    def prior_sample(self, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        next(generator)

    def prior_distribution(self, traces=1000, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        ret = []
        time_start = time.time()
        for i in range(traces):
            if (util.verbosity > 1):
                duration = time.time() - time_start
                print('                                                                \r{} | {} | {} / {} | {:,.2f} traces/s'.format(util.days_hours_mins_secs_str(duration), util.progress_bar(i+1, traces), i+1, traces, i / duration), end='\r')
                sys.stdout.flush()
            ret.append(next(generator))
        if (util.verbosity > 1):
            print()
        return Empirical(ret)

    def posterior_distribution(self, traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, *args, **kwargs):
        if (inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK) and (self._inference_network is None):
            print('Warning: Cannot run inference with inference network because there is none available. Use learn_inference_network first.')
            print('Warning: Running with InferenceEngine.IMPORTANCE_SAMPLING')
            inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            self._inference_network.eval()
            traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD_USE_INFERENCE_NETWORK, proposal_network=self._inference_network, *args, **kwargs)
        else:
            traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD_IMPORTANCE, proposal_network=None, *args, **kwargs)
        log_weights = [trace.log_prob for trace in traces]
        results = [trace.result for trace in traces]
        return Empirical(results, log_weights)

    def lmh_posterior(self, mixin=10000, num_samples=1000, *args, **kwargs):

        old_trace = self._single_trace(*args, **kwargs)
        samples = []
        i = 0

        time_start = time.time()
        while len(samples) < num_samples:
            log_pdf_old = old_trace.log_y() 
            resample_index = random.randint(0, len(old_trace.samples) - 1)   
            resample_address = old_trace.samples[resample_index].address
            new_trace = self._continue_trace_at_address(resample_address, old_trace, *args, **kwargs)
            log_pdf_old += new_trace.log_fresh()
            log_pdf_new = new_trace.log_y() + old_trace.log_fresh()


            accept_ratio = (log_pdf_new - log_pdf_old).data[0]

            duration = time.time() - time_start
            if math.log(random.uniform(0,1)) < accept_ratio:
                old_trace = new_trace
                if i > mixin:
                    samples.append(new_trace)
                i += 1
                print('                                                                \r{} | {} | {} / {} | {:,.2f} traces/s'.format(util.days_hours_mins_secs_str(duration), util.progress_bar(i+1, num_samples + mixin), i+1, num_samples + mixin, i / duration), end='\r')
        return samples

    def learn_inference_network(self, lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_reshape=None, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=64, batch_size=64, valid_size=256, learning_rate=0.001, weight_decay=1e-4, early_stop_traces=-1, use_trace_cache=False, *args, **kwargs):
        if self._inference_network is None:
            print('Creating new inference network...')
            traces = self._prior_traces(valid_size, trace_mode=TraceMode.RECORD_TRAIN_INFERENCE_NETWORK, *args, **kwargs)
            valid_batch = Batch(traces)
            self._inference_network = InferenceNetwork(model_name=self.name, lstm_dim=lstm_dim, lstm_depth=lstm_depth, observe_embedding=observe_embedding, observe_reshape=observe_reshape, observe_embedding_dim=observe_embedding_dim, sample_embedding=sample_embedding, sample_embedding_dim=sample_embedding_dim, address_embedding_dim=address_embedding_dim, valid_batch=valid_batch, cuda=util._cuda_enabled)
            self._inference_network.polymorph()
        else:
            print('Continuing to train existing inference network...')

        optimizer = optim.Adam(self._inference_network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_trace_cache and self._trace_cache_path is None:
            print('Warning: There is no trace cache assigned, training with online trace generation.')
            use_trace_cache = False

        if use_trace_cache:
            def new_batch_func():
                current_files = self._trace_cache_current_files()
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

                while len(self._trace_cache) < batch_size:
                    current_file = random.choice(current_files)
                    new_traces = self._load_traces(current_file)
                    random.shuffle(new_traces)
                    self._trace_cache += new_traces

                traces = self._trace_cache[0:batch_size]
                self._trace_cache[0:batch_size] = []
                return Batch(traces)
        else:
            def new_batch_func():
                traces = self._prior_traces(batch_size, trace_mode=TraceMode.RECORD_TRAIN_INFERENCE_NETWORK, *args, **kwargs)
                return Batch(traces)

        self._inference_network.train()
        self._inference_network.optimize(new_batch_func, optimizer, early_stop_traces)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network.save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetwork.load(file_name, util._cuda_enabled, util._cuda_device)

    def trace_length_mean(self, traces=1000, *args, **kwargs):
        traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return trace_length_dist.mean

    def trace_length_stddev(self, traces=1000, *args, **kwargs):
        traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return trace_length_dist.stddev

    def trace_length_min(self, traces=1000, *args, **kwargs):
        traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return min(trace_length_dist.values_numpy)

    def trace_length_max(self, traces=1000, *args, **kwargs):
        traces = self._prior_traces(traces, trace_mode=TraceMode.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return max(trace_length_dist.values_numpy)

    def save_trace_cache(self, trace_cache_path, files=16, traces_per_file=512, *args, **kwargs):
        f = 0
        done = False
        while not done:
            traces = self._prior_traces(traces_per_file, trace_mode=TraceMode.RECORD_TRAIN_INFERENCE_NETWORK, proposal_network=None, *args, **kwargs)
            file_name = os.path.join(trace_cache_path, str(uuid.uuid4()))
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
            raise RuntimeError('Cannot load trace cache.')

        # print('Loading trace cache of length {}'.format(data['length']))
        if data['model_name'] != self.name:
            print(colored('Warning: different model names (loaded traces: {}, current model: {})'.format(data['model_name'], self.name), 'red', attrs=['bold']))
        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded traces: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded traces: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        return data['traces']


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
