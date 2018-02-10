import torch.nn as nn

from .distributions import Empirical
from . import state, util
from .state import TraceState
from .nn import ObserveEmbedding, SampleEmbedding, Batch, ProposalNetwork

class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._proposal_network = None

    def forward(self):
        raise NotImplementedError()

    def _prior_trace_generator(self, trace_state=TraceState.RECORD, *args, **kwargs):
        while True:
            state.begin_trace(self.forward, trace_state)
            res = self.forward(*args, **kwargs)
            trace = state.end_trace()
            trace.set_result(res)
            yield trace

    def _prior_sample_generator(self, *args, **kwargs):
        while True:
            yield self.forward(*args, **kwargs)

    def _prior_traces(self, samples=10, trace_state=TraceState.RECORD, *args, **kwargs):
        generator = self._prior_trace_generator(trace_state, *args, **kwargs)
        return [next(generator) for i in range(samples)]

    def prior_sample(self, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        next(generator)

    def prior_distribution(self, samples=1000, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        return Empirical([next(generator) for i in range(samples)])

    def posterior_distribution(self, samples=1000, learned_proposal=False, *args, **kwargs):
        if learned_proposal and (self._proposal_network is None):
            print('Warning: Cannot run inference with learned proposal because there is no proposal network trained')
            learned_proposal = False
        if learned_proposal:
            traces = self._prior_traces(samples, trace_state=TraceState.RECORD_USE_PROPOSAL, *args, **kwargs)
        else:
            traces = self._prior_traces(samples, trace_state=TraceState.RECORD, *args, **kwargs)
        log_weights = [trace.log_prob for trace in traces]
        results = [trace.result for trace in traces]
        return Empirical(results, log_weights)

    def learn_proposal(self, lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=64, valid_size=256, *args, **kwargs):
        if self._proposal_network is None:
            print('Creating new proposal network...')
            traces = self._prior_traces(valid_size, trace_state=TraceState.RECORD_LEARN_PROPOSAL, *args, **kwargs)
            valid_batch = Batch(traces)
            self._proposal_network = ProposalNetwork(self.name, lstm_dim, lstm_depth, observe_embedding, observe_embedding_dim, sample_embedding, sample_embedding_dim, address_embedding_dim, valid_batch)
            self._proposal_network.polymorph()
