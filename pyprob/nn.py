import enum
import math
import random
import gc
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored

from . import util
from .distributions import Categorical, Normal


class ObserveEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class SampleEmbedding(enum.Enum):
    FULLY_CONNECTED = 0


class Batch(object):
    def __init__(self, traces, sort=True):
        self.batch = traces
        self.length = len(traces)
        self.traces_lengths = []
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
        self.traces_lengths = [t.length for t in self.batch]

    def __getitem__(self, key):
        return self.batch[key]

    def __setitem__(self, key, value):
        self.batch[key] = value

    def sort_by_observes_length(self):
        return Batch(sorted(self.batch, reverse=True, key=lambda x:x.observes_variable.nelement()), False)


class ObserveEmbeddingFC(nn.Module):
    def __init__(self, input_example_non_batch, output_dim):
        super().__init__()
        self._input_dim = input_example_non_batch.nelement()
        self._lin1 = nn.Linear(self._input_dim, self._input_dim)
        self._lin2 = nn.Linear(self._input_dim, output_dim)
        self._lin3 = nn.Linear(output_dim, output_dim)
        nn.init.xavier_uniform(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = F.relu(self._lin1(x.view(-1, self._input_dim)))
        x = F.relu(self._lin2(x))
        x = F.relu(self._lin3(x))
        return x


class SampleEmbeddingFC(nn.Module):
    def __init__(self, input_dim, output_dim, input_is_one_hot_index=False, input_one_hot_dim=None):
        super().__init__()
        self._input_is_one_hot_index = input_is_one_hot_index
        self._input_one_hot_dim = input_one_hot_dim
        if input_is_one_hot_index:
            if input_dim != 1:
                raise ValueError('If input_is_one_hot_index=True, input_dim should be 1 (the index of one-hot value in a vector of length input_one_hot_dim)')
            input_dim = input_one_hot_dim
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, output_dim)
        self._lin3 = nn.Linear(output_dim, output_dim)
        nn.init.xavier_uniform(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        if self._input_is_one_hot_index:
            x = torch.stack([util.one_hot(self._input_one_hot_dim, int(v)) for v in x])
        x = F.relu(self._lin1(x))
        x = F.relu(self._lin2(x))
        x = F.relu(self._lin3(x))
        return x


class ProposalNormal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, 2)
        nn.init.xavier_uniform(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = self._lin2(x)
        means = x[:,0].unsqueeze(1)
        stddevs = x[:,1].unsqueeze(1)
        stddevs = nn.Softplus()(stddevs)
        prior_means = torch.cat([s.distribution.mean for s in samples])
        prior_stddevs = torch.cat([s.distribution.stddev for s in samples])
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return Normal(means, stddevs)


class ProposalCategorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._lin1 = nn.Linear(input_dim, input_dim)
        self._lin2 = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform(self._lin1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform(self._lin2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x, samples):
        x = F.relu(self._lin1(x))
        x = F.softmax(self._lin2(x), dim=1)
        return Categorical(probs=x)


class InferenceNetwork(nn.Module):
    def __init__(self, model_name='Unnamed model', lstm_dim=512, lstm_depth=2, observe_embedding=ObserveEmbedding.FULLY_CONNECTED, observe_embedding_dim=512, sample_embedding=SampleEmbedding.FULLY_CONNECTED, sample_embedding_dim=32, address_embedding_dim=64, valid_batch=None, cuda=False, device=None):
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
        self._cuda = cuda
        self._cuda_device = device

        self._state_new_trace = True
        self._state_observes = None
        self._state_observes_embedding = None
        self._state_lstm_hidden_state = None

        self._address_embeddings = {}
        self._distribution_type_embeddings = {}

        self._address_embedding_empty = util.to_variable(torch.zeros(self._address_embedding_dim))
        self._distribution_type_embedding_empty = util.to_variable(torch.zeros(self._distribution_type_embedding_dim))

        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        example_observes = self._valid_batch[0].observes_variable # To do: we need to check that all the observes in the batch, to be more intelligent
        if self._observe_embedding == ObserveEmbedding.FULLY_CONNECTED:
            self._observe_embedding_layer = ObserveEmbeddingFC(example_observes, self._observe_embedding_dim)
        else:
            raise ValueError('Unknown observation embedding: {}'.format(self._observe_embedding))
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
                        if isinstance(distribution, Categorical):
                            sample_embedding_layer = SampleEmbeddingFC(sample.value.nelement(), self._sample_embedding_dim, input_is_one_hot_index=True, input_one_hot_dim=distribution.length)
                        else:
                            sample_embedding_layer = SampleEmbeddingFC(sample.value.nelement(), self._sample_embedding_dim)
                    else:
                        raise ValueError('Unkown sample embedding: {}'.format(self._sample_embedding))

                    if isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self._lstm_dim)
                    elif isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategorical(self._lstm_dim, distribution.length)
                    else:
                        raise ValueError('Unsupported distribution: {}'.format(distribution.name))

                    self._sample_embedding_layers[address] = sample_embedding_layer
                    self._proposal_layers[address] = proposal_layer
                    self.add_module('sample_embedding_layer({})'.format(address), sample_embedding_layer)
                    self.add_module('proposal_layer({})'.format(address), proposal_layer)
                    print('Polymorphing, new layers for address: {}'.format(address))
                    layers_changed = True

        if self._cuda:
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
            self._state_lstm_hidden_state = (h0, h0)
            self._state_new_trace = False
        else:
            prev_address = previous_sample.address_suffixed
            prev_distribution = previous_sample.distribution
            prev_value = previous_sample.value
            if prev_address in self._sample_embedding_layers:
                prev_sample_embedding = self._sample_embedding_layers[prev_address](prev_value.unsqueeze(0).float())
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

        current_address = current_sample.address_suffixed
        current_distribution = current_sample.distribution
        if not current_address in self._proposal_layers:
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

    def loss(self, batch, optimizer):
        gc.collect()

        obs = torch.stack([b.observes_variable for b in batch])
        obs_emb = self._observe_embedding_layer(obs)
        for b in range(batch.length):
            batch[b].observes_embedding = obs_emb[b]

        for sub_batch in batch.sub_batches:
            sub_batch_length = len(sub_batch)
            example_trace = sub_batch[0]

            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address_suffixed
                current_distribution = current_sample.distribution
                current_addres_embedding = self._address_embeddings[current_address]
                current_distribution_type_embedding = self._distribution_type_embeddings[current_distribution.name]

                if time_step == 0:
                    prev_sample_embedding = util.to_variable(torch.zeros(sub_batch_length, self._sample_embedding_dim))
                    prev_addres_embedding = self._address_embedding_empty
                    prev_distribution_type_embedding = self._distribution_type_embedding_empty
                else:
                    prev_sample = example_trace.samples[time_step - 1]
                    prev_address = prev_sample.address_suffixed
                    prev_distribution = prev_sample.distribution
                    smp = torch.stack([trace.samples[time_step - 1].value.float() for trace in sub_batch])
                    prev_sample_embedding = self._sample_embedding_layers[prev_address](smp)
                    prev_addres_embedding = self._address_embeddings[prev_address]
                    prev_distribution_type_embedding = self._distribution_type_embeddings[prev_distribution.name]

                for b in range(sub_batch_length):
                    t = torch.cat([sub_batch[b].observes_embedding,
                                   prev_sample_embedding[b],
                                   prev_distribution_type_embedding,
                                   prev_addres_embedding,
                                   current_distribution_type_embedding,
                                   current_addres_embedding])
                    sub_batch[b].samples[time_step].lstm_input = t

        lstm_input = []
        for time_step in range(batch.traces_max_length):
            t = []
            for b in range(batch.length):
                trace = batch[b]
                if time_step < trace.length:
                    t.append(trace.samples[time_step].lstm_input)
                else:
                    t.append(util.to_variable(torch.zeros(self._lstm_input_dim)))
            t = torch.cat(t).view(batch.length, -1)
            lstm_input.append(t)
        lstm_input = torch.cat(lstm_input).view(batch.traces_max_length, batch.length, -1)
        lstm_input = nn.utils.rnn.pack_padded_sequence(lstm_input, batch.traces_lengths)

        h0 = util.to_variable(torch.zeros(self._lstm_depth, batch.length, self._lstm_dim))
        lstm_output, _ = self._lstm(lstm_input, (h0, h0))
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)

        for b in range(batch.length):
            trace = batch[b]
            for time_step in range(trace.length):
                trace.samples[time_step].lstm_output = lstm_output[time_step, b]

        log_prob = 0
        for sub_batch in batch.sub_batches:
            sub_batch_length = len(sub_batch)
            example_trace = sub_batch[0]

            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address_suffixed

                p = []
                for b in range(sub_batch_length):
                    p.append(sub_batch[b].samples[time_step].lstm_output)
                proposal_input = torch.cat(p).view(sub_batch_length, -1)

                current_samples = [trace.samples[time_step] for trace in sub_batch]
                current_samples_values = torch.stack([s.value for s in current_samples])
                proposal_distribution = self._proposal_layers[current_address](proposal_input, current_samples)
                l = proposal_distribution.log_prob(current_samples_values)
                if util.has_nan_or_inf(l):
                    return False, 0
                log_prob += torch.sum(l)

        loss = -log_prob / batch.length

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return True, float(loss)

    def optimize(self, new_batch_func, optimizer, early_stop_traces=-1):
        iteration = 0
        trace = 0
        loss_initial = None
        loss_min = float('inf')
        loss_max = None
        loss_prev = float('inf')
        stop = False
        print('Trace     | Init. loss | Max. loss  | Min. loss  | Curr. loss')
        while not stop:
            iteration += 1
            batch = new_batch_func()
            self.polymorph(batch)
            success, loss = self.loss(batch, optimizer)
            if not success:
                print('Cannot compute loss, stopping. Loss: {}'.format(loss))
                stop = True
            else:
                if loss_initial is None:
                    loss_initial = loss
                    loss_initial_str = '{:+.3e}'.format(loss_initial)
                    loss_max = loss
                    loss_max_str = '{:+.3e}'.format(loss_max)
                if loss < loss_min:
                    loss_min = loss
                    loss_str = colored('{:+.3e}'.format(loss), 'green', attrs=['bold'])
                    loss_min_str = colored('{:+.3e}'.format(loss_min), 'green', attrs=['bold'])
                elif loss > loss_max:
                    loss_max = loss
                    loss_str = colored('{:+.3e}'.format(loss), 'red', attrs=['bold'])
                    loss_max_str = colored('{:+.3e}'.format(loss_max), 'red', attrs=['bold'])
                else:
                    if loss < loss_prev:
                        loss_str = colored('{:+.3e}'.format(loss), 'green')
                    elif loss > loss_prev:
                        loss_str = colored('{:+.3e}'.format(loss), 'red')
                    else:
                        loss_str = '{:+.3e}'.format(loss)
                    loss_min_str = '{:+.3e}'.format(loss_min)
                    loss_max_str = '{:+.3e}'.format(loss_max)
                loss_prev = loss
                trace += batch.length
                trace_str = '{:9}'.format('{:,}'.format(trace))
                if early_stop_traces != -1:
                    if trace >= early_stop_traces:
                        stop = True

                print('{} | {} | {} | {} | {}\r'.format(trace_str, loss_initial_str, loss_max_str, loss_min_str, loss_str), end='')
