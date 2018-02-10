import torch.nn as nn

from .distributions import Empirical
from . import state

class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        self.name = name

    def forward(self):
        raise NotImplementedError()

    def _prior_trace_generator(self, *args, **kwargs):
        while True:
            state.begin_trace(self.forward)
            res = self.forward(*args, **kwargs)
            trace = state.end_trace()
            trace.set_result(res)
            yield trace

    def _prior_sample_generator(self, *args, **kwargs):
        while True:
            yield self.forward(*args, **kwargs)

    def prior_sample(self, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        next(generator)

    def prior_distribution(self, samples=1000, *args, **kwargs):
        generator = self._prior_sample_generator(*args, **kwargs)
        return Empirical([next(generator) for i in range(samples)])

    def posterior_distribution(self, samples=1000, *args, **kwargs):
        generator = self._prior_trace_generator(*args, **kwargs)
        traces = [next(generator) for i in range(samples)]
        log_weights = [trace.log_prob for trace in traces]
        results = [trace.result for trace in traces]
        return Empirical(results, log_weights)
