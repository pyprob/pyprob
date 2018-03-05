import torch.nn as nn
import torch.optim as optim
import time

from .distributions import Empirical
from . import state, util
from .state import TraceState
from .nn import ObserveEmbedding, SampleEmbedding, Batch, InferenceNetwork
from .remote import ModelServer


class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._inference_network = None

    def forward(self):
        raise NotImplementedError()

    def _prior_trace_generator(self, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs):
        while True:
            if trace_state == TraceState.RECORD_USE_INFERENCE_NETWORK:
                self._inference_network.new_trace(util.pack_observes_to_variable(kwargs['observation']).unsqueeze(0))
            state.begin_trace(self.forward, trace_state, proposal_network)
            res = self.forward(*args, **kwargs)
            trace = state.end_trace()
            trace.set_result(res)
            yield trace

    def _prior_sample_generator(self, *args, **kwargs):
        while True:
            yield self.forward(*args, **kwargs)

    def _prior_traces(self, samples=10, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs):
        generator = self._prior_trace_generator(trace_state, proposal_network, *args, **kwargs)
        ret = []
        time_start = time.time()
        for i in range(samples):
            print('                                                                \r{} | {} / {} | {:,} traces/s'.format(util.progress_bar(i+1, samples), i+1, samples, int(i / (time.time() - time_start))), end='\r')
            ret.append(next(generator))
        return ret

    def prior_sample(self, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        next(generator)

    def prior_distribution(self, samples=1000, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        ret = []
        time_start = time.time()
        for i in range(samples):
            print('                                                                \r{} | {} / {} | {:,} traces/s'.format(util.progress_bar(i+1, samples), i+1, samples, int(i / (time.time() - time_start))), end='\r')
            ret.append(next(generator))
        return Empirical(ret)

    def posterior_distribution(self, samples=1000, use_inference_network=False, *args, **kwargs):
        if use_inference_network and (self._inference_network is None):
            print('Warning: Cannot run inference with inference network because there is none available. Use learn_inference_network first.')
            use_inference_network = False
        if use_inference_network:
            self._inference_network.eval()
            traces = self._prior_traces(samples, trace_state=TraceState.RECORD_USE_INFERENCE_NETWORK, proposal_network=self._inference_network, *args, **kwargs)
        else:
            traces = self._prior_traces(samples, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs)
        log_weights = [trace.log_prob for trace in traces]
        results = [trace.result for trace in traces]
        return Empirical(results, log_weights)

    def learn_inference_network(self, lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=64, batch_size=64, valid_size=256, learning_rate=0.001, weight_decay=1e-4, early_stop_traces=-1, auto_save_file_name='', *args, **kwargs):
        if self._inference_network is None:
            print('Creating new inference network...')
            traces = self._prior_traces(valid_size, trace_state=TraceState.RECORD_TRAIN_INFERENCE_NETWORK, *args, **kwargs)
            valid_batch = Batch(traces)
            self._inference_network = InferenceNetwork(model_name=self.name, lstm_dim=lstm_dim, lstm_depth=lstm_depth, observe_embedding=observe_embedding, observe_embedding_dim=observe_embedding_dim, sample_embedding=sample_embedding, sample_embedding_dim=sample_embedding_dim, address_embedding_dim=address_embedding_dim, valid_batch=valid_batch, cuda=util._cuda_enabled)
            self._inference_network.polymorph()
        else:
            print('Continuing to train existing inference network...')

        optimizer = optim.Adam(self._inference_network.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def new_batch_func():
            traces = self._prior_traces(batch_size, trace_state=TraceState.RECORD_TRAIN_INFERENCE_NETWORK, *args, **kwargs)
            return Batch(traces)

        self._inference_network.train()
        self._inference_network.optimize(new_batch_func, optimizer, early_stop_traces)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network.save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetwork.load(file_name, util._cuda_enabled, util._cuda_device)

    def trace_length_mean(self, samples=1000, *args, **kwargs):
        traces = self._prior_traces(samples, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return trace_length_dist.mean

    def trace_length_stddev(self, samples=1000, *args, **kwargs):
        traces = self._prior_traces(samples, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return trace_length_dist.stddev

    def trace_length_min(self, samples=1000, *args, **kwargs):
        traces = self._prior_traces(samples, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return min(trace_length_dist.values_numpy)

    def trace_length_max(self, samples=1000, *args, **kwargs):
        traces = self._prior_traces(samples, trace_state=TraceState.RECORD, proposal_network=None, *args, **kwargs)
        trace_length_dist = Empirical([trace.length for trace in traces])
        return max(trace_length_dist.values_numpy)


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
