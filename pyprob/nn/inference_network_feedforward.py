import torch
import torch.nn as nn
import torch.optim as optim

from .. import TraceMode


class Batch():
    def __init__(self, traces):
        self.batch = traces
        self.length = len(traces)
        sub_batches = {}
        for trace in traces:
            if trace.length == 0:
                raise ValueError('Trace of length zero.')
            trace_hash = ''.join([variable.address for variable in trace.variables_controlled])
            if trace_hash not in sub_batches:
                sub_batches[trace_hash] = []
            sub_batches[trace_hash].append(trace)
        self.sub_batches = list(sub_batches.values())


class InferenceNetworkFeedForward(nn.Module):
    def __init__(self, model, prior_inflation, valid_batch_size=64):
        super().__init__()
        self._model = model
        self._prior_inflation = prior_inflation
        self._valid_batch = self.get_batch(valid_batch_size)
        self._optimizer = None

    def get_batch(self, length=64, *args, **kwargs):
        traces = self._model._traces(length, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, *args, **kwargs)
        return Batch(traces)

    def polymorph(self):
        return False

    def optimize(self, num_traces=None, batch_size=64, *args, **kwargs):
        iteration = 0
        stop = False
        while not stop:
            iteration += 1
            batch = self.get_batch(batch_size)
            layers_changed = self.polymorph(batch)

            if (self._optimizer is None) or layers_changed:
                self._optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-8)
