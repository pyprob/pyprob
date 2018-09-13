import torch
import torch.nn as nn
import torch.optim as optim
import gc

from . import EmbeddingFeedForward, ProposalNormal
from .. import util, TraceMode, ObserveEmbedding
from ..distributions import Normal


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
    # observe_template example: {'obs': {'input_shape': torch.Size([100]), 'output_shape': torch.Size([25]), 'observe_embedding':ObserveEmbedding.FULLY_CONNECTED}}
    def __init__(self, model, prior_inflation, valid_batch_size=64, observe_template={}):
        super().__init__()
        self._model = model
        self._prior_inflation = prior_inflation
        self._layer_observe_embedding = {}
        self._layer_proposal = {}
        self._layer_hidden_shape = None
        self._infer_observe = None
        self._infer_observe_embedding = {}
        self._optimizer = None
        self._valid_batch = self.get_batch(valid_batch_size)
        self._init_layer_observe_embeddings(observe_template)
        self._polymorph(self._valid_batch)

    def _init_layer_observe_embeddings(self, observe_template):
        if len(observe_template) == 0:
            raise ValueError('At least one observe is needed to initialize inference network.')
        self._layer_observe_embeddings = {}
        observe_embedding_total_dim = 0
        for name, value in observe_template:
            if 'input_shape' in value:
                input_shape = value['input_shape']
            else:
                raise ValueError('Expecting the input_shape of variable: {}'.format(name))
            if 'output_shape' in value:
                output_shape = value['output_shape']
            else:
                output_shape = torch.Size([int(util.prod(input_shape)/2)])
                print('output_shape not provided, using {} for variable: {}'.format(output_shape, name))
            if 'observe_embedding' in value:
                observe_embedding = value['observe_embedding']
            else:
                print('Observe embedding not provided, using default FULLY_CONNECTED for variable: {}'.format(name))
                observe_embedding = ObserveEmbedding.FULLY_CONNECTED
            if observe_embedding == ObserveEmbedding.FULLY_CONNECTED:
                self._layer_observe_embeddings[name] = EmbeddingFeedForward(input_shape=input_shape, output_shape=output_shape)
            else:
                raise ValueError('Unknown observe_embedding: {}'.format(observe_embedding))
            observe_embedding_total_dim += util.prod(output_shape)
            self._layer_hidden_shape = torch.Size([observe_embedding_total_dim])

    def _get_batch(self, length=64, *args, **kwargs):
        traces = self._model._traces(length, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, *args, **kwargs)
        return Batch(traces)

    def _embed_observe(self, observe=None):
        if observe is None:
            raise ValueError('All observes in observe_template are needed to initialize a new trace.')
        embedding = {}
        for name, layer in self._layer_observe_embedding:
            embedding[name] = layer.forward(observe[name])
        return embedding

    def infer_trace_init(self, observe=None):
        self._infer_observe = observe
        self._infer_observe_embedding = self._embed_observe(observe)

    def infer_trace_step(self):
        return

    def _polymorph(self, batch):
        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for variable in example_trace.variables_controlled:
                address = variable.address
                distribution = variable.distribution
                variable_shape = variable.value.shape
                if address not in self._layer_proposal:
                    print('New proposal layer for address: {}'.format(util.truncate_str(address)))
                    if isinstance(distribution, Normal):
                        layer = ProposalNormal(self._layer_hidden_shape, variable_shape)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    self._layer_proposal[address] = layer
                    self.add_module('layer_proposal({})'.format(address), layer)
                    layers_changed = True
        return layers_changed

    def _loss(self, batch):
        gc.collect()
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            sub_batch_length = len(sub_batch)
            embedding = torch.stack([self._embed_observe(trace.named_variables) for trace in sub_batch])
            log_prob = 0.
            for time_step in range(example_trace.length_controlled):
                variable = example_trace.variables_controlled[time_step]

    def optimize(self, num_traces=None, batch_size=64, *args, **kwargs):
        iteration = 0
        stop = False
        while not stop:
            iteration += 1
            batch = self._get_batch(batch_size)
            layers_changed = self._polymorph(batch)

            if (self._optimizer is None) or layers_changed:
                self._optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-8)
