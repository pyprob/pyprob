import enum
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util
from .distributions import Normal


class ObserveEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class SampleEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class Batch(object):
    def __init__(self, traces, sort=True):
        self.batch = traces
        self.length = len(traces)
        self.traces_max_length = 0
        self.observes_max_length = 0
        sb = {}
        for trace in traces:
            if trace.length is None:
                raise ValueError('Trace of length zero')
            if trace.length > self.traces_max_length:
                self.traces_max_length = trace.length
            if trace.observes_variable.size(0) > self.observes_max_length:
                self.observes_max_length = trace.observes_variable.size(0)
            h = hash(trace.addresses_suffixed())
            if not h in sb:
                sb[h] = []
            sb[h].append(trace)
        self.sub_batches = []
        for _, t in sb.items():
            self.sub_batches.append(t)
        if sort:
            # Sort the batch in descreasing trace length
            self.batch = sorted(self.batch, reverse=True, key=lambda t: t.length)

    def sort_by_observes_length(self):
        return Batch(sorted(self.batch, reverse=True, key=lambda x:x.observes_variable.nelement()), False)


class SampleEmbeddingFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
        nn.init.xavier_uniform(self.lin1.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x


class ProposalNormal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.lin2 = nn.Linear(input_dim, 2)
        nn.init.xavier_uniform(self.lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self.lin2.weight)

    def forward(self, x, samples):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        means = x[:,0].unsqueeze(1)
        stddevs = x[:,1].unsqueeze(1)
        stddevs = nn.Softplus()(stddevs)
        prior_means = util.to_variable(util.Tensor([s.distribution.prior_mean for s in samples]))
        prior_stddevs = util.to_variable(util.Tensor([s.distribution.prior_stddev for s in samples]))
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return Normal(means, stddevs)


class ProposalNetwork(nn.Module):
    def __init__(self, model_name='Unnamed model', lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=64, valid_batch=None):
        super().__init__()
        self._model_name = model_name
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._observe_embedding = observe_embedding
        self._observe_embedding_dim = observe_embedding_dim
        self._sample_embedding = sample_embedding
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = 1 # Needs to match the number of distribution types in pyprob (except Emprical)
        self._valid_batch = valid_batch

        self._address_embeddings = {}
        self._distribution_type_embeddings = {}

        self._address_embedding_empty = util.to_variable(torch.zeros(self._address_embedding_dim))
        self._distribution_type_embedding_empty = util.to_variable(torch.zeros(self._distribution_type_embedding_dim))

        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        self._sample_embedding_layers = {}
        self._proposal_layers = {}

    def _add_address(self, address):
        if not address in self._address_embeddings:
            print('Polymorphing, new address: {}'.format(address))
            i = len(self._address_embeddings)
            if i < self._address_embedding_dim:
                t = util.one_hot(self._address_embedding_dim, i)
                self._address_embeddings[address] = t
            else:
                print('Warning: overflow (collision) in address embeddings. Allowed: {}; Encountered: {}'.format(self._address_embedding_dim, i + 1))
                self._address_embeddings[address] = random.choice(list(self._address_embeddings.values()))

    def _add_distribution_type(self, distribution_type):
        if not distribution_type in self._distribution_type_embeddings:
            print('Polymorphing, new distribution type: {}'.format(distribution_type))
            i = len(self._distribution_type_embeddings)
            if i < self._distribution_type_embedding_dim:
                t = util.one_hot(self._distribution_type_embedding_dim, i)
                self._distribution_type_embeddings[distribution_type] = t
            else:
                print('Warning: overflow (collision) in distribution type embeddings. Allowed: {}; Encountered: {}'.format(self._distribution_type_embedding_dim, i + 1))
                self._distribution_type_embeddings[distribution_type] = random.choice(list(self._distribution_type_embeddings.values()))

    def polymorph(self, batch=None):
        if batch is None:
            if self._valid_batch is None:
                return
            else:
                batch = self._valid_batch

        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]

            for sample in example_trace.samples:
                address = sample.address_suffixed
                distribution = sample.distribution

                # Update the dictionaries for address and distribution type embeddings
                self._add_address(address)
                self._add_distribution_type(distribution.name)

                if not address in self._sample_embedding_layers:
                    if self._sample_embedding == SampleEmbedding.FULLY_CONNECTED:
                        sample_embedding_layer = SampleEmbeddingFC(sample.value.nelement(), self._sample_embedding_dim)
                    else:
                        raise ValueError('Unkown sample embedding: {}'.format(self._sample_embedding))

                    if isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self._lstm_dim)
                    else:
                        raise ValueError('Unsupported distribution: {}'.format(distribution.name))

                    self._sample_embedding_layers[address] = sample_embedding_layer
                    self._proposal_layers[address] = proposal_layer
                    self.add_module('sample_embedding_layer({})'.format(address), sample_embedding_layer)
                    self.add_module('proposal_layer({})'.format(address), proposal_layer)
                    print('Polymorphing: new layers for address: {}'.format(address))
                    layers_changed = True
        return layers_changed
