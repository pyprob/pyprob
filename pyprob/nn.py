import random
import gc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from termcolor import colored
from threading import Thread
import time
import tempfile
import uuid
import shutil
import os
import tarfile
from collections import OrderedDict

from . import util, __version__, ObserveEmbedding, SampleEmbedding, Optimizer, TrainingObservation
from .distributions import Categorical, Mixture, Normal, TruncatedNormal, Uniform, Poisson, Kumaraswamy


class Batch(object):
    def __init__(self, traces, sort=True):
        self.batch = traces
        self.length = len(traces)
        self.traces_lengths = []
        self.traces_max_length = 0
        sb = {}
        for trace in traces:
            if trace.length == 0:
                raise ValueError('Trace of length zero')
            if trace.length > self.traces_max_length:
                self.traces_max_length = trace.length
            h = hash(trace.addresses())
            if h not in sb:
                sb[h] = []
            sb[h].append(trace)
        self.sub_batches = []
        for _, t in sb.items():
            self.sub_batches.append(t)
        if sort:
            # Sort the batch in descreasing trace length
            self.batch = sorted(self.batch, reverse=True, key=lambda t: t.length)
        self.traces_lengths = [t.length for t in self.batch]

    def __getitem__(self, key):
        return self.batch[key]

    def __setitem__(self, key, value):
        self.batch[key] = value

    def cuda(self, device):
        for trace in self.batch:
            trace.cuda(device)

    def cpu(self):
        for trace in self.batch:
            trace.cpu()

    def sort_by_observes_length(self):
        return Batch(sorted(self.batch, reverse=True, key=lambda x: x._inference_network_training_observes_variable.nelement()), False)


class ObserveEmbeddingFC(nn.Module):
    def __init__(self, input_example_non_batch, output_dim):
        super().__init__()
        self._input_dim = input_example_non_batch.nelement()
        self._lin1 = nn.Linear(self._input_dim, self._input_dim)
        # self._drop1 = nn.Dropout()
        self._lin2 = nn.Linear(self._input_dim, output_dim)
        # self._lin3 = nn.Linear(output_dim, output_dim)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self._lin3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = F.relu(self._lin1(x.view(-1, self._input_dim)))
        # x = self._drop1(x)
        x = F.relu(self._lin2(x))
        # x = F.relu(self._lin3(x))
        return x


class ObserveEmbeddingConvNet2D5C(nn.Module):
    def __init__(self, input_example_non_batch, output_dim, reshape=None):
        super().__init__()
        self._reshape = reshape
        if self._reshape is not None:
            input_example_non_batch = input_example_non_batch.view(self._reshape)
            self._reshape.insert(0, -1)  # For correct handling of the batch dimension in self.forward
        if input_example_non_batch.dim() == 2:
            self._input_sample = input_example_non_batch.unsqueeze(0).cpu()
        elif input_example_non_batch.dim() == 3:
            self._input_sample = input_example_non_batch.cpu()
        else:
            raise RuntimeError('ObserveEmbeddingConvNet2D5C: Expecting a 3d input_example_non_batch (num_channels x height x width) or a 2d input_example_non_batch (height x width). Received: {}'.format(input_example_non_batch.size()))
        self._input_channels = self._input_sample.size(0)
        self._output_dim = output_dim
        self._conv1 = nn.Conv2d(self._input_channels, 64, 3)
        self._conv2 = nn.Conv2d(64, 64, 3)
        self._conv3 = nn.Conv2d(64, 128, 3)
        self._conv4 = nn.Conv2d(128, 128, 3)
        self._conv5 = nn.Conv2d(128, 128, 3)

    def configure(self):
        self._cnn_output_dim = self._forward_cnn(self._input_sample.unsqueeze(0)).view(-1).size(0)
        self._lin1 = nn.Linear(self._cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = F.relu(self._conv3(x))
        x = F.relu(self._conv4(x))
        x = F.relu(self._conv5(x))
        x = nn.MaxPool2d(2)(x)
        return x

    def forward(self, x):
        if self._reshape is not None:
            x = x.view(self._reshape)
        if x.dim() == 3:  # This indicates that there are no channel dimensions and we have BxHxW
            x = x.unsqueeze(1)  # Add a channel dimension of 1 after the batch dimension so thhat we have BxCxHxW
        x = self._forward_cnn(x)
        x = x.view(-1, self._cnn_output_dim)
        x = F.relu(self._lin1(x))
        x = F.relu(self._lin2(x))
        return x


class ObserveEmbeddingConvNet3D4C(nn.Module):
    def __init__(self, input_example_non_batch, output_dim, reshape=None):
        super().__init__()
        self._reshape = reshape
        if self._reshape is not None:
            input_example_non_batch = input_example_non_batch.view(self._reshape)
            self._reshape.insert(0, -1)  # For correct handling of the batch dimension in self.forward
        if input_example_non_batch.dim() == 3:
            self._input_sample = input_example_non_batch.unsqueeze(0).cpu()
        elif input_example_non_batch.dim() == 4:
            self._input_sample = input_example_non_batch.cpu()
        else:
            raise RuntimeError('ObserveEmbeddingConvNet3D4C: Expecting a 4d input_example_non_batch (num_channels x depth x height x width) or a 3d input_example_non_batch (depth x height x width). Received: {}'.format(input_example_non_batch.size()))
        self._input_channels = self._input_sample.size(0)
        self._output_dim = output_dim
        self._conv1 = nn.Conv3d(self._input_channels, 64, 3)
        self._conv2 = nn.Conv3d(64, 64, 3)
        self._conv3 = nn.Conv3d(64, 128, 3)
        self._conv4 = nn.Conv3d(128, 128, 3)

    def configure(self):
        self._cnn_output_dim = self._forward_cnn(self._input_sample.unsqueeze(0)).view(-1).size(0)
        self._lin1 = nn.Linear(self._cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = nn.MaxPool3d(2)(x)
        x = F.relu(self._conv3(x))
        x = F.relu(self._conv4(x))
        x = nn.MaxPool3d(2)(x)
        return x

    def forward(self, x):
        if self._reshape is not None:
            x = x.view(self._reshape)
        if x.dim() == 4:  # This indicates that there are no channel dimensions and we have BxDxHxW
            x = x.unsqueeze(1)  # Add a channel dimension of 1 after the batch dimension so that we have BxCxDxHxW
        x = self._forward_cnn(x)
        x = x.view(-1, self._cnn_output_dim)
        x = F.relu(self._lin1(x))
        x = F.relu(self._lin2(x))
        return x


class SampleEmbeddingFC(nn.Module):
    def __init__(self, input_dim, output_dim, input_is_one_hot_index=False, input_one_hot_dim=None):
        super().__init__()
        self._input_dim = input_dim
        self._input_is_one_hot_index = input_is_one_hot_index
        self._input_one_hot_dim = input_one_hot_dim
        if input_is_one_hot_index:
            if input_dim != 1:
                raise ValueError('If input_is_one_hot_index=True, input_dim should be 1 (the index of one-hot value in a vector of length input_one_hot_dim)')
            input_dim = input_one_hot_dim
        self._lin1 = nn.Linear(input_dim, int(output_dim/2))
        self._lin2 = nn.Linear(int(output_dim/2), output_dim)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self._input_is_one_hot_index:
            x = torch.stack([util.one_hot(self._input_one_hot_dim, int(v)) for v in x])
        else:
            x = x.view(-1, self._input_dim)
        x = F.relu(self._lin1(x))
        x = F.relu(self._lin2(x))
        return x


class ProposalCategorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._drop1 = nn.Dropout()
        self._lin2 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._drop1(x)
        x = F.softmax(self._lin2(x), dim=1) + util._epsilon
        return Categorical(probs=x)


class ProposalNormal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._output_dim = output_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, self._output_dim * 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, 0:self._output_dim]
        stddevs = x[:, self._output_dim:2*self._output_dim]
        stddevs = nn.Softplus()(stddevs)
        prior_means = torch.stack([s.distribution.mean for s in samples])
        prior_stddevs = torch.stack([s.distribution.stddev for s in samples])
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return Normal(means, stddevs)


class ProposalUniform(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._lin1 = nn.Linear(input_dim, int(input_dim/2))
        self._lin2 = nn.Linear(int(input_dim/2), 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        # prior_means = util.to_variable(torch.stack([s.distribution.mean[0] for s in samples]))
        prior_stddevs = util.to_variable(torch.stack([s.distribution.stddev[0] for s in samples]))
        prior_lows = util.to_variable(torch.stack([s.distribution.low for s in samples]))
        prior_highs = util.to_variable(torch.stack([s.distribution.high for s in samples]))
        means = x[:, 0].unsqueeze(1)
        means = prior_lows + F.sigmoid(means) * (prior_highs - prior_lows)
        stddevs = x[:, 1].unsqueeze(1)
        stddevs = util._epsilon + F.sigmoid(stddevs) * prior_stddevs
        return TruncatedNormal(means, stddevs, prior_lows, prior_highs)


class ProposalUniformMixture(nn.Module):
    def __init__(self, input_dim, mixture_components=5):
        super().__init__()
        self._mixture_components = mixture_components
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, 3 * self._mixture_components)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, 0:self._mixture_components]
        stddevs = x[:, self._mixture_components:2*self._mixture_components]
        coeffs = x[:, 2*self._mixture_components:3*self._mixture_components]
        stddevs = F.softplus(stddevs)
        coeffs = F.softmax(coeffs, dim=1)
        prior_means = util.to_variable(torch.stack([s.distribution.mean[0] for s in samples]))
        prior_stddevs = util.to_variable(torch.stack([s.distribution.stddev[0] for s in samples]))
        prior_lows = util.to_variable(torch.stack([s.distribution.low[0] for s in samples]))
        prior_highs = util.to_variable(torch.stack([s.distribution.high[0] for s in samples]))
        prior_means = prior_means.expand_as(means)
        prior_stddevs = prior_stddevs.expand_as(means)

        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        distributions = [TruncatedNormal(means[:, i:i+1], stddevs[:, i:i+1], prior_lows, prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)


class ProposalUniformKumaraswamyMixture(nn.Module):
    def __init__(self, input_dim, mixture_components=10):
        super().__init__()
        self._mixture_components = mixture_components
        self._lin1 = nn.Linear(input_dim, int(input_dim/2))
        self._lin2 = nn.Linear(int(input_dim/2), 3 * self._mixture_components)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        shape1s = x[:, 0:self._mixture_components]
        shape1s = 2. + F.relu(shape1s)
        shape2s = x[:, self._mixture_components:2*self._mixture_components]
        shape2s = 1. + F.relu(shape2s)
        coeffs = x[:, 2*self._mixture_components:3*self._mixture_components]
        coeffs = F.softmax(coeffs, dim=1)
        # prior_means = util.to_variable(torch.stack([s.distribution.mean[0] for s in samples]))
        # prior_stddevs = util.to_variable(torch.stack([s.distribution.stddev[0] for s in samples]))
        prior_lows = util.to_variable(torch.stack([s.distribution.low for s in samples]))
        prior_highs = util.to_variable(torch.stack([s.distribution.high for s in samples]))
        # means = prior_means + (means * prior_stddevs)
        # stddevs = stddevs * prior_stddevs
        # return TruncatedNormal(means, stddevs, prior_lows, prior_highs)
        distributions = [Kumaraswamy(shape1s[:, i:i+1], shape2s[:, i:i+1], prior_lows, prior_highs) for i in range(self._mixture_components)]
        # dist = Kumaraswamy(shape1s, shape2s, low=prior_lows, high=prior_highs)
        # print('dist', dist)
        return Mixture(distributions, coeffs)


class ProposalUniformKumaraswamy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._lin1 = nn.Linear(input_dim, int(input_dim/2))
        self._lin2 = nn.Linear(int(input_dim/2), 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        shape1s = x[:, 0].unsqueeze(1)
        shape1s = 2. + F.relu(shape1s)
        shape2s = x[:, 1].unsqueeze(1)
        shape2s = 1. + F.relu(shape2s)
        # prior_means = util.to_variable(torch.stack([s.distribution.mean[0] for s in samples]))
        # prior_stddevs = util.to_variable(torch.stack([s.distribution.stddev[0] for s in samples]))
        prior_lows = util.to_variable(torch.stack([s.distribution.low for s in samples]))
        prior_highs = util.to_variable(torch.stack([s.distribution.high for s in samples]))
        # means = prior_means + (means * prior_stddevs)
        # stddevs = stddevs * prior_stddevs
        # return TruncatedNormal(means, stddevs, prior_lows, prior_highs)
        dist = Kumaraswamy(shape1s, shape2s, low=prior_lows, high=prior_highs)
        # print('dist', dist)
        return dist


class ProposalPoisson(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, 2)
        nn.init.xavier_uniform_(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    # TODO: implement a better proposal for Poisson
    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:, 0].unsqueeze(1)
        stddevs = x[:, 1].unsqueeze(1)
        stddevs = F.softplus(stddevs)
        prior_means = util.to_variable(torch.stack([s.distribution.mean[0] for s in samples]))
        prior_stddevs = util.to_variable(torch.stack([s.distribution.stddev[0] for s in samples]))
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return TruncatedNormal(means, stddevs, 0, 40)


class InferenceNetworkSimple(nn.Module):
    def __init__(self, model_name='Unnamed model', observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_reshape=None, observe_embedding_dim=512, valid_batch=None):
        super().__init__()
        self._model_name = model_name
        self._observe_embedding = observe_embedding
        self._observe_reshape = observe_reshape
        self._observe_embedding_dim = observe_embedding_dim
        self._valid_batch = valid_batch
        self._on_cuda = util._cuda_enabled
        self._cuda_device = util._cuda_device
        self._trained_on = 'CPU'
        self._total_train_seconds = 0
        self._total_train_traces = 0
        self._total_train_iterations = 0
        self._loss_initial = None
        self._loss_min = float('inf')
        self._loss_max = None
        self._loss_previous = float('inf')
        self._history_train_loss = []
        self._history_train_loss_trace = []
        self._history_valid_loss = []
        self._history_valid_loss_trace = []
        self._history_num_params = []
        self._history_num_params_trace = []
        self._created = util.get_time_str()
        self._modified = util.get_time_str()
        self._updates = 0
        self._pyprob_version = __version__
        self._torch_version = torch.__version__
        self._optimizer_type = ''

        self._state_new_trace = True
        self._state_observes = None
        self._state_observes_embedding = None

        self._address_stats = OrderedDict()
        self._trace_stats = OrderedDict()

        example_observes = self._valid_batch[0].pack_observes()  # To do: we need to check all the observes in the batch, to be more intelligent
        if self._observe_embedding == ObserveEmbedding.FULLY_CONNECTED:
            self._observe_embedding_layer = ObserveEmbeddingFC(example_observes, self._observe_embedding_dim)
        elif self._observe_embedding == ObserveEmbedding.CONVNET_2D_5C:
            self._observe_embedding_layer = ObserveEmbeddingConvNet2D5C(example_observes, self._observe_embedding_dim, self._observe_reshape)
            self._observe_embedding_layer.configure()
        elif self._observe_embedding == ObserveEmbedding.CONVNET_3D_4C:
            self._observe_embedding_layer = ObserveEmbeddingConvNet3D4C(example_observes, self._observe_embedding_dim, self._observe_reshape)
            self._observe_embedding_layer.configure()
        else:
            raise ValueError('Unknown observation embedding: {}'.format(self._observe_embedding))
        # self._sample_embedding_layers = {}
        self._proposal_layers = {}

    def cuda(self, device=None):
        self._on_cuda = True
        self._cuda_device = device
        super().cuda(device)
        self._valid_batch.cuda(device)
        return self

    def cpu(self):
        self._on_cuda = False
        super().cpu()
        self._valid_batch.cpu()
        return self

    def _update_stats(self, batch):
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            # Update address stats
            for sample in example_trace._samples_all:
                address = sample.address
                if address not in self._address_stats:
                    address_id = 'A' + str(len(self._address_stats) + 1)
                    self._address_stats[address] = [len(sub_batch), address_id, sample.distribution.name, sample.control, sample.replace, sample.observed]
                else:
                    self._address_stats[address][0] += len(sub_batch)
            # Update trace stats
            trace_str = example_trace.addresses()
            if trace_str not in self._trace_stats:
                trace_id = 'T' + str(len(self._trace_stats) + 1)

                # TODO: the following is not needed when the Trace code is updated to keep track of samples_controlled, samples_controlled_observed
                samples_controlled_observed = []
                replaced_indices = []
                for i in range(len(example_trace._samples_all)):
                    sample = example_trace._samples_all[i]
                    if sample.control and i not in replaced_indices:
                        if sample.replace:
                            for j in range(i + 1, len(example_trace._samples_all)):
                                if example_trace._samples_all[j].address_base == sample.address_base:
                                    # example_trace.samples_replaced.append(sample)
                                    sample = example_trace._samples_all[j]
                                    replaced_indices.append(j)
                        samples_controlled_observed.append(sample)
                    elif sample.observed:
                        samples_controlled_observed.append(sample)
                trace_addresses = [self._address_stats[sample.address][1] for sample in samples_controlled_observed]
                self._trace_stats[trace_str] = [len(sub_batch), trace_id, len(example_trace._samples_all),  len(example_trace.samples), trace_addresses]
            else:
                self._trace_stats[trace_str][0] += len(sub_batch)

    def polymorph(self, batch=None):
        if batch is None:
            if self._valid_batch is None:
                return
            else:
                batch = self._valid_batch

        self._update_stats(batch)

        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]

            for sample in example_trace.samples:
                address = sample.address
                distribution = sample.distribution

                if address not in self._proposal_layers:
                    print('New layers for address ({}): {}'.format(self._address_stats[address][1], util.truncate_str(address)))

                    if isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategorical(self._observe_embedding_dim, distribution.length_categories)
                    elif isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self._observe_embedding_dim, distribution.length_variates)
                    elif isinstance(distribution, Uniform):
                        proposal_layer = ProposalUniformKumaraswamyMixture(self._observe_embedding_dim)
                    elif isinstance(distribution, Poisson):
                        proposal_layer = ProposalPoisson(self._observe_embedding_dim)
                    else:
                        raise ValueError('Unsupported distribution: {}'.format(distribution.name))

                    self._proposal_layers[address] = proposal_layer
                    self.add_module('proposal_layer({})'.format(address), proposal_layer)
                    layers_changed = True

        if layers_changed:
            num_params = 0
            for p in self.parameters():
                num_params += p.nelement()
            print('New trainable params: {:,}'.format(num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)

        if self._on_cuda:
            self.cuda(self._cuda_device)

        return layers_changed

    def new_trace(self, observes=None):
        self._state_new_trace = True
        self._state_observes = observes
        self._state_observes_embedding = self._observe_embedding_layer.forward(observes)

    def forward_one_time_step(self, previous_sample, current_sample):
        if self._state_observes is None:
            raise RuntimeError('Cannot run the inference network without observations. Call new_trace and supply observations.')

        success = True
        current_address = current_sample.address
        current_distribution = current_sample.distribution
        if current_address not in self._proposal_layers:
            print('Warning: no proposal layer for: {}'.format(current_address))
            success = False

        if success:
            proposal_distribution = self._proposal_layers[current_address](self._state_observes_embedding, [current_sample])
            return proposal_distribution
        else:
            print('Warning: no proposal could be made, prior will be used')
            return current_distribution

    def loss(self, batch, optimizer=None, training_observation=TrainingObservation.OBSERVE_DIST_SAMPLE):
        gc.collect()

        batch_loss = 0
        for sub_batch in batch.sub_batches:
            obs = torch.stack([trace.pack_observes(training_observation) for trace in sub_batch])
            obs_emb = self._observe_embedding_layer(obs)

            sub_batch_length = len(sub_batch)
            example_trace = sub_batch[0]

            log_prob = 0
            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address

                current_samples = [trace.samples[time_step] for trace in sub_batch]
                current_samples_values = torch.stack([s.value for s in current_samples])
                proposal_distribution = self._proposal_layers[current_address](obs_emb, current_samples)
                l = proposal_distribution.log_prob(current_samples_values)
                if util.has_nan_or_inf(l):
                    print(colored('Warning: NaN, -Inf, or Inf encountered in proposal log_prob.', 'red', attrs=['bold']))
                    print('proposal_distribution', proposal_distribution)
                    print('current_samples_values', current_samples_values)
                    print('log_prob', l)
                    print('Fixing -Inf')
                    l = util.replace_negative_inf(l)
                    print('log_prob', l)
                    if util.has_nan_or_inf(l):
                        print(colored('Nan or Inf present in proposal log_prob.', 'red', attrs=['bold']))
                        return False, 0
                log_prob += util.safe_torch_sum(l)

            sub_batch_loss = -log_prob / sub_batch_length
            batch_loss += float(sub_batch_loss * sub_batch_length)

            if optimizer is not None:
                optimizer.zero_grad()
                sub_batch_loss.backward()
                optimizer.step()

        return True, batch_loss / batch.length

    def optimize(self, new_batch_func, training_observation, optimizer_type, num_traces, learning_rate, momentum, weight_decay, valid_interval, auto_save, auto_save_file_name):
        self._trained_on = 'CUDA' if util._cuda_enabled else 'CPU'
        self._optimizer_type = optimizer_type
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time.time()
        time_last_batch = time.time()
        iteration = 0
        trace = 0
        last_validation_trace = -valid_interval + 1
        stop = False
        print('Train. time | Trace     | Init. loss| Min. loss | Curr. loss| T.since min | Traces/sec')
        max_print_line_len = 0

        while not stop:
            iteration += 1
            batch = new_batch_func()
            layers_changed = self.polymorph(batch)

            if iteration == 1 or layers_changed:
                if optimizer_type == Optimizer.ADAM:
                    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else:  # optimizer_type == Optimizer.SGD:
                    optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
            success, loss = self.loss(batch, optimizer, training_observation)
            if not success:
                print(colored('Cannot compute loss, skipping batch. Loss: {}'.format(loss), 'red', attrs=['bold']))
            else:
                if self._loss_initial is None:
                    self._loss_initial = loss
                    self._loss_max = loss
                loss_initial_str = '{:+.2e}'.format(self._loss_initial)
                # loss_max_str = '{:+.3e}'.format(self._loss_max)
                if loss < self._loss_min:
                    self._loss_min = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'green', attrs=['bold'])
                    loss_min_str = colored('{:+.2e}'.format(self._loss_min), 'green', attrs=['bold'])
                    time_loss_min = time.time()
                    time_since_loss_min_str = colored(util.days_hours_mins_secs_str(0), 'green', attrs=['bold'])
                elif loss > self._loss_max:
                    self._loss_max = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'red', attrs=['bold'])
                    # loss_max_str = colored('{:+.3e}'.format(self._loss_max), 'red', attrs=['bold'])
                else:
                    if loss < self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'green')
                    elif loss > self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'red')
                    else:
                        loss_str = '{:+.2e}'.format(loss)
                    loss_min_str = '{:+.2e}'.format(self._loss_min)
                    # loss_max_str = '{:+.3e}'.format(self._loss_max)
                    time_since_loss_min_str = util.days_hours_mins_secs_str(time.time() - time_loss_min)

                self._loss_previous = loss
                self._total_train_iterations += 1
                trace += batch.length
                self._total_train_traces += batch.length
                total_training_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                self._total_train_seconds = prev_total_train_seconds + (time.time() - time_start)
                total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                traces_per_second_str = '{:,.1f}'.format(int(batch.length / (time.time() - time_last_batch)))
                time_last_batch = time.time()
                if num_traces != -1:
                    if trace >= num_traces:
                        stop = True

                self._history_train_loss.append(loss)
                self._history_train_loss_trace.append(self._total_train_traces)
                if trace - last_validation_trace > valid_interval:
                    print('\rComputing validation loss...', end='\r')
                    _, valid_loss = self.loss(self._valid_batch)
                    self._history_valid_loss.append(valid_loss)
                    self._history_valid_loss_trace.append(self._total_train_traces)
                    last_validation_trace = trace - 1
                    if auto_save:
                        file_name = auto_save_file_name + '_' + util.get_time_stamp()
                        print('\rSaving to disk...', end='\r')
                        self._save(file_name)

                print_line = '{} | {} | {} | {} | {} | {} | {}'.format(total_training_seconds_str, total_training_traces_str, loss_initial_str, loss_min_str, loss_str, time_since_loss_min_str, traces_per_second_str)
                max_print_line_len = max(len(print_line), max_print_line_len)
                print(print_line.ljust(max_print_line_len), end='\r')
                sys.stdout.flush()
        print()

    def _save(self, file_name):
        self._modified = util.get_time_str()
        self._updates += 1

        data = {}
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__
        data['inference_network'] = self

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_inference_network')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_inference_network')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def _load(file_name, cuda=False, device=None):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_inference_network')
            tar.extract('pyprob_inference_network', tmp_dir)
            tar.close()
            if cuda:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
                shutil.rmtree(tmp_dir)
        except:
            raise RuntimeError('Cannot load inference network.')

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded network: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded network: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        ret = data['inference_network']
        if cuda:
            if ret._on_cuda:
                if ret._cuda_device != device:
                    print(colored('Warning: loading CUDA (device {}) network to CUDA (device {})'.format(ret._cuda_device, device), 'red', attrs=['bold']))
                    ret.cuda(device)
            else:
                print(colored('Warning: loading CPU network to CUDA (device {})'.format(device), 'red', attrs=['bold']))
                ret.cuda(device)
        else:
            if ret._on_cuda:
                print(colored('Warning: loading CUDA (device {}) network to CPU'.format(ret._cuda_device), 'red', attrs=['bold']))
                ret.cpu()
        return ret


class InferenceNetworkLSTM(nn.Module):
    def __init__(self, model_name='Unnamed model', lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_reshape=None, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=256, valid_batch=None):
        super().__init__()
        self._model_name = model_name
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._observe_embedding = observe_embedding
        self._observe_reshape = observe_reshape
        self._observe_embedding_dim = observe_embedding_dim
        self._sample_embedding = sample_embedding
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = 4  # Needs to match the number of distribution types in pyprob (except Emprical)
        self._valid_batch = valid_batch
        self._on_cuda = util._cuda_enabled
        self._cuda_device = util._cuda_device
        self._trained_on = 'CPU'
        self._total_train_seconds = 0
        self._total_train_traces = 0
        self._total_train_iterations = 0
        self._loss_initial = None
        self._loss_min = float('inf')
        self._loss_max = None
        self._loss_previous = float('inf')
        self._history_train_loss = []
        self._history_train_loss_trace = []
        self._history_valid_loss = []
        self._history_valid_loss_trace = []
        self._history_num_params = []
        self._history_num_params_trace = []
        self._created = util.get_time_str()
        self._modified = util.get_time_str()
        self._updates = 0
        self._pyprob_version = __version__
        self._torch_version = torch.__version__
        self._optimizer_type = ''

        self._state_new_trace = True
        self._state_observes = None
        self._state_observes_embedding = None
        self._state_lstm_hidden_state = None

        self._address_stats = OrderedDict()
        self._trace_stats = OrderedDict()

        self._address_embeddings = {}
        self._distribution_type_embeddings = {}

        self._address_embedding_empty = util.to_variable(torch.zeros(self._address_embedding_dim))
        self._distribution_type_embedding_empty = util.to_variable(torch.zeros(self._distribution_type_embedding_dim))

        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        example_observes = self._valid_batch[0].pack_observes()  # To do: we need to check all the observes in the batch, to be more intelligent
        if self._observe_embedding == ObserveEmbedding.FULLY_CONNECTED:
            self._observe_embedding_layer = ObserveEmbeddingFC(example_observes, self._observe_embedding_dim)
        elif self._observe_embedding == ObserveEmbedding.CONVNET_2D_5C:
            self._observe_embedding_layer = ObserveEmbeddingConvNet2D5C(example_observes, self._observe_embedding_dim, self._observe_reshape)
            self._observe_embedding_layer.configure()
        elif self._observe_embedding == ObserveEmbedding.CONVNET_3D_4C:
            self._observe_embedding_layer = ObserveEmbeddingConvNet3D4C(example_observes, self._observe_embedding_dim, self._observe_reshape)
            self._observe_embedding_layer.configure()
        else:
            raise ValueError('Unknown observation embedding: {}'.format(self._observe_embedding))
        self._sample_embedding_layers = {}
        self._proposal_layers = {}

    def cuda(self, device=None):
        self._on_cuda = True
        self._cuda_device = device
        super().cuda(device)
        self._address_embedding_empty = self._address_embedding_empty.cuda(device)
        self._distribution_type_embedding_empty = self._distribution_type_embedding_empty.cuda(device)
        for k, t in self._address_embeddings.items():
            self._address_embeddings[k] = t.cuda(device)
        for k, t in self._distribution_type_embeddings.items():
            self._distribution_type_embeddings[k] = t.cuda(device)
        self._valid_batch.cuda(device)
        return self

    def cpu(self):
        self._on_cuda = False
        super().cpu()
        self._address_embedding_empty = self._address_embedding_empty.cpu()
        self._distribution_type_embedding_empty = self._distribution_type_embedding_empty.cpu()
        for k, t in self._address_embeddings.items():
            self._address_embeddings[k] = t.cpu()
        for k, t in self._distribution_type_embeddings.items():
            self._distribution_type_embeddings[k] = t.cpu()
        self._valid_batch.cpu()
        return self

    def _update_stats(self, batch):
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            # Update address stats
            for sample in example_trace._samples_all:
                address = sample.address
                if address not in self._address_stats:
                    address_id = 'A' + str(len(self._address_stats) + 1)
                    self._address_stats[address] = [len(sub_batch), address_id, sample.distribution.name, sample.control, sample.replace, sample.observed]
                else:
                    self._address_stats[address][0] += len(sub_batch)
            # Update trace stats
            trace_str = example_trace.addresses()
            if trace_str not in self._trace_stats:
                trace_id = 'T' + str(len(self._trace_stats) + 1)

                # TODO: the following is not needed when the Trace code is updated to keep track of samples_controlled, samples_controlled_observed
                samples_controlled_observed = []
                replaced_indices = []
                for i in range(len(example_trace._samples_all)):
                    sample = example_trace._samples_all[i]
                    if sample.control and i not in replaced_indices:
                        if sample.replace:
                            for j in range(i + 1, len(example_trace._samples_all)):
                                if example_trace._samples_all[j].address_base == sample.address_base:
                                    # example_trace.samples_replaced.append(sample)
                                    sample = example_trace._samples_all[j]
                                    replaced_indices.append(j)
                        samples_controlled_observed.append(sample)
                    elif sample.observed:
                        samples_controlled_observed.append(sample)
                trace_addresses = [self._address_stats[sample.address][1] for sample in samples_controlled_observed]
                self._trace_stats[trace_str] = [len(sub_batch), trace_id, len(example_trace._samples_all),  len(example_trace.samples), trace_addresses]
            else:
                self._trace_stats[trace_str][0] += len(sub_batch)

    def _add_address(self, address):
        if address not in self._address_embeddings:
            t = Parameter(util.Tensor(self._address_embedding_dim).normal_())
            self._address_embeddings[address] = t
            self.register_parameter('address_embedding_' + address, t)
            print('New layers for address ({}): {}'.format(self._address_stats[address][1], util.truncate_str(address)))

    def _add_distribution_type(self, distribution_type):
        if distribution_type not in self._distribution_type_embeddings:
            print('New distribution type: {}'.format(distribution_type))
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

        self._update_stats(batch)

        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]

            for sample in example_trace.samples:
                address = sample.address
                distribution = sample.distribution

                # Update the dictionaries for address and distribution type embeddings
                self._add_address(address)
                self._add_distribution_type(distribution.name)

                if address not in self._sample_embedding_layers:
                    if self._sample_embedding == SampleEmbedding.FULLY_CONNECTED:
                        if isinstance(distribution, Categorical):
                            sample_embedding_layer = SampleEmbeddingFC(sample.value.nelement(), self._sample_embedding_dim, input_is_one_hot_index=True, input_one_hot_dim=distribution.length_categories)
                        else:
                            sample_embedding_layer = SampleEmbeddingFC(sample.value.nelement(), self._sample_embedding_dim)
                    else:
                        raise ValueError('Unkown sample embedding: {}'.format(self._sample_embedding))

                    if isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategorical(self._lstm_dim, distribution.length_categories)
                    elif isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self._lstm_dim, distribution.length_variates)
                    elif isinstance(distribution, Uniform):
                        proposal_layer = ProposalUniformKumaraswamyMixture(self._lstm_dim)
                    elif isinstance(distribution, Poisson):
                        proposal_layer = ProposalPoisson(self._lstm_dim)
                    else:
                        raise ValueError('Unsupported distribution: {}'.format(distribution.name))

                    self._sample_embedding_layers[address] = sample_embedding_layer
                    self._proposal_layers[address] = proposal_layer
                    self.add_module('sample_embedding_layer({})'.format(address), sample_embedding_layer)
                    self.add_module('proposal_layer({})'.format(address), proposal_layer)
                    layers_changed = True

        if layers_changed:
            num_params = 0
            for p in self.parameters():
                num_params += p.nelement()
            print('New trainable params: {:,}'.format(num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)

        if self._on_cuda:
            self.cuda(self._cuda_device)

        return layers_changed

    def new_trace(self, observes=None):
        self._state_new_trace = True
        self._state_observes = observes
        self._state_observes_embedding = self._observe_embedding_layer.forward(observes)

    def forward_one_time_step(self, previous_sample, current_sample):
        if self._state_observes is None:
            raise RuntimeError('Cannot run the inference network without observations. Call new_trace and supply observations.')

        success = True
        if self._state_new_trace:
            prev_sample_embedding = util.to_variable(torch.zeros(1, self._sample_embedding_dim))
            prev_addres_embedding = self._address_embedding_empty
            prev_distribution_type_embedding = self._distribution_type_embedding_empty
            h0 = util.to_variable(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            c0 = util.to_variable(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            self._state_lstm_hidden_state = (h0, c0)
            self._state_new_trace = False
        else:
            prev_address = previous_sample.address
            prev_distribution = previous_sample.distribution
            prev_value = previous_sample.value
            if prev_value.dim() == 0:
                prev_value = prev_value.unsqueeze(0)
            if prev_address in self._sample_embedding_layers:
                prev_sample_embedding = self._sample_embedding_layers[prev_address](prev_value.float())
            else:
                print('Warning: no sample embedding layer for: {}'.format(prev_address))
                success = False
            if prev_address in self._address_embeddings:
                prev_addres_embedding = self._address_embeddings[prev_address]
            else:
                print('Warning: unknown address (previous): {}'.format(prev_address))
                success = False
            if prev_distribution.name in self._distribution_type_embeddings:
                prev_distribution_type_embedding = self._distribution_type_embeddings[prev_distribution.name]
            else:
                print('Warning: unkown distribution type (previous): {}'.format(prev_distribution.name))
                success = False

        current_address = current_sample.address
        # print(current_address)
        current_distribution = current_sample.distribution
        if current_address not in self._proposal_layers:
            print('Warning: no proposal layer for: {}'.format(current_address))
            success = False
        if current_address in self._address_embeddings:
            current_addres_embedding = self._address_embeddings[current_address]
        else:
            print('Warning: unknown address (current): {}'.format(current_address))
            success = False
        if current_distribution.name in self._distribution_type_embeddings:
            current_distribution_type_embedding = self._distribution_type_embeddings[current_distribution.name]
        else:
            print('Warning: unkown distribution type (current): {}'.format(current_distribution.name))
            success = False

        if success:
            t = [self._state_observes_embedding[0],
                 prev_sample_embedding[0],
                 prev_distribution_type_embedding,
                 prev_addres_embedding,
                 current_distribution_type_embedding,
                 current_addres_embedding]
            t = torch.cat(t).unsqueeze(0)
            lstm_input = t.unsqueeze(0)
            lstm_output, self._state_lstm_hidden_state = self._lstm(lstm_input, self._state_lstm_hidden_state)
            proposal_input = lstm_output[0]
            proposal_distribution = self._proposal_layers[current_address](proposal_input, [current_sample])
            return proposal_distribution
        else:
            print('Warning: no proposal could be made, prior will be used')
            return current_distribution

    def loss(self, batch, optimizer=None, training_observation=TrainingObservation.OBSERVE_DIST_SAMPLE):
        gc.collect()

        batch_loss = 0
        for sub_batch in batch.sub_batches:
            obs = torch.stack([trace.pack_observes(training_observation) for trace in sub_batch])
            obs_emb = self._observe_embedding_layer(obs)

            sub_batch_length = len(sub_batch)
            example_trace = sub_batch[0]

            lstm_input = []
            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address
                current_distribution = current_sample.distribution
                current_addres_embedding = self._address_embeddings[current_address]
                current_distribution_type_embedding = self._distribution_type_embeddings[current_distribution.name]

                if time_step == 0:
                    prev_sample_embedding = util.to_variable(torch.zeros(sub_batch_length, self._sample_embedding_dim))
                    prev_addres_embedding = self._address_embedding_empty
                    prev_distribution_type_embedding = self._distribution_type_embedding_empty
                else:
                    prev_sample = example_trace.samples[time_step - 1]
                    prev_address = prev_sample.address
                    prev_distribution = prev_sample.distribution
                    smp = torch.stack([trace.samples[time_step - 1].value.float() for trace in sub_batch])
                    prev_sample_embedding = self._sample_embedding_layers[prev_address](smp)
                    prev_addres_embedding = self._address_embeddings[prev_address]
                    prev_distribution_type_embedding = self._distribution_type_embeddings[prev_distribution.name]

                lstm_input_time_step = []
                for b in range(sub_batch_length):
                    t = torch.cat([obs_emb[b],
                                   prev_sample_embedding[b],
                                   prev_distribution_type_embedding,
                                   prev_addres_embedding,
                                   current_distribution_type_embedding,
                                   current_addres_embedding])
                    lstm_input_time_step.append(t)
                lstm_input.append(torch.stack(lstm_input_time_step))

            lstm_input = torch.stack(lstm_input)

            h0 = util.to_variable(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            c0 = util.to_variable(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            lstm_output, _ = self._lstm(lstm_input, (h0, c0))

            log_prob = 0
            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address

                proposal_input = lstm_output[time_step]

                current_samples = [trace.samples[time_step] for trace in sub_batch]
                current_samples_values = torch.stack([s.value for s in current_samples])
                proposal_distribution = self._proposal_layers[current_address](proposal_input, current_samples)
                l = proposal_distribution.log_prob(current_samples_values)
                if util.has_nan_or_inf(l):
                    print(colored('Warning: NaN, -Inf, or Inf encountered in proposal log_prob.', 'red', attrs=['bold']))
                    print('proposal_distribution', proposal_distribution)
                    print('current_samples_values', current_samples_values)
                    print('log_prob', l)
                    print('Fixing -Inf')
                    l = util.replace_negative_inf(l)
                    print('log_prob', l)
                    if util.has_nan_or_inf(l):
                        print(colored('Nan or Inf present in proposal log_prob.', 'red', attrs=['bold']))
                        return False, 0
                log_prob += util.safe_torch_sum(l)

            sub_batch_loss = -log_prob / sub_batch_length
            batch_loss += float(sub_batch_loss * sub_batch_length)

            if optimizer is not None:
                optimizer.zero_grad()
                sub_batch_loss.backward()
                optimizer.step()

        return True, batch_loss / batch.length

    def optimize(self, new_batch_func, training_observation, optimizer_type, num_traces, learning_rate, momentum, weight_decay, valid_interval, auto_save, auto_save_file_name):
        self._trained_on = 'CUDA' if util._cuda_enabled else 'CPU'
        self._optimizer_type = optimizer_type
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time.time()
        time_last_batch = time.time()
        iteration = 0
        trace = 0
        last_validation_trace = -valid_interval + 1
        stop = False
        print('Train. time | Trace     | Init. loss| Min. loss | Curr. loss| T.since min | Traces/sec')
        max_print_line_len = 0

        while not stop:
            iteration += 1

            batch = new_batch_func()
            layers_changed = self.polymorph(batch)

            if iteration == 1 or layers_changed:
                if optimizer_type == Optimizer.ADAM:
                    optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else:  # optimizer_type == Optimizer.SGD:
                    optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
            success, loss = self.loss(batch, optimizer, training_observation)
            if not success:
                print(colored('Cannot compute loss, skipping batch. Loss: {}'.format(loss), 'red', attrs=['bold']))
            else:
                if self._loss_initial is None:
                    self._loss_initial = loss
                    self._loss_max = loss
                loss_initial_str = '{:+.2e}'.format(self._loss_initial)
                # loss_max_str = '{:+.3e}'.format(self._loss_max)
                if loss < self._loss_min:
                    self._loss_min = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'green', attrs=['bold'])
                    loss_min_str = colored('{:+.2e}'.format(self._loss_min), 'green', attrs=['bold'])
                    time_loss_min = time.time()
                    time_since_loss_min_str = colored(util.days_hours_mins_secs_str(0), 'green', attrs=['bold'])
                elif loss > self._loss_max:
                    self._loss_max = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'red', attrs=['bold'])
                    # loss_max_str = colored('{:+.3e}'.format(self._loss_max), 'red', attrs=['bold'])
                else:
                    if loss < self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'green')
                    elif loss > self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'red')
                    else:
                        loss_str = '{:+.2e}'.format(loss)
                    loss_min_str = '{:+.2e}'.format(self._loss_min)
                    # loss_max_str = '{:+.3e}'.format(self._loss_max)
                    time_since_loss_min_str = util.days_hours_mins_secs_str(time.time() - time_loss_min)

                self._loss_previous = loss
                self._total_train_iterations += 1
                trace += batch.length
                self._total_train_traces += batch.length
                total_training_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                self._total_train_seconds = prev_total_train_seconds + (time.time() - time_start)
                total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                traces_per_second_str = '{:,.1f}'.format(int(batch.length / (time.time() - time_last_batch)))
                time_last_batch = time.time()
                if num_traces != -1:
                    if trace >= num_traces:
                        stop = True

                self._history_train_loss.append(loss)
                self._history_train_loss_trace.append(self._total_train_traces)
                if trace - last_validation_trace > valid_interval:
                    print('\rComputing validation loss...', end='\r')
                    _, valid_loss = self.loss(self._valid_batch)
                    self._history_valid_loss.append(valid_loss)
                    self._history_valid_loss_trace.append(self._total_train_traces)
                    last_validation_trace = trace - 1
                    if auto_save:
                        file_name = auto_save_file_name + '_' + util.get_time_stamp()
                        print('\rSaving to disk...', end='\r')
                        self._save(file_name)

                print_line = '{} | {} | {} | {} | {} | {} | {}'.format(total_training_seconds_str, total_training_traces_str, loss_initial_str, loss_min_str, loss_str, time_since_loss_min_str, traces_per_second_str)
                max_print_line_len = max(len(print_line), max_print_line_len)
                print(print_line.ljust(max_print_line_len), end='\r')
                sys.stdout.flush()
        print()

    def _save(self, file_name):
        self._modified = util.get_time_str()
        self._updates += 1

        data = {}
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__
        data['inference_network'] = self

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_inference_network')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_inference_network')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def _load(file_name, cuda=False, device=None):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_inference_network')
            tar.extract('pyprob_inference_network', tmp_dir)
            tar.close()
            if cuda:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
                shutil.rmtree(tmp_dir)
        except:
            raise RuntimeError('Cannot load inference network.')

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded network: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded network: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        ret = data['inference_network']
        if cuda:
            if ret._on_cuda:
                if ret._cuda_device != device:
                    print(colored('Warning: loading CUDA (device {}) network to CUDA (device {})'.format(ret._cuda_device, device), 'red', attrs=['bold']))
                    ret.cuda(device)
            else:
                print(colored('Warning: loading CPU network to CUDA (device {})'.format(device), 'red', attrs=['bold']))
                ret.cuda(device)
        else:
            if ret._on_cuda:
                print(colored('Warning: loading CUDA (device {}) network to CPU'.format(ret._cuda_device), 'red', attrs=['bold']))
                ret.cpu()
        return ret
