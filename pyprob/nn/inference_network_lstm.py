import torch
import torch.nn as nn
from termcolor import colored

from . import InferenceNetwork, EmbeddingFeedForward, ProposalNormalNormalMixture, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, ProposalPoissonTruncatedNormalMixture
from .. import util
from ..distributions import Normal, Uniform, Categorical, Poisson


class InferenceNetworkLSTM(InferenceNetwork):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FEEDFORWARD, 'reshape': [10, 10], 'dim': 32, 'depth': 2}}
    def __init__(self, lstm_dim=512, lstm_depth=1, sample_embedding_dim=4, address_embedding_dim=64, distribution_type_embedding_dim=8, proposal_mixture_components=10, *args, **kwargs):
        super().__init__(network_type='InferenceNetworkLSTM', *args, **kwargs)
        self._layers_proposal = nn.ModuleDict()
        self._layers_sample_embedding = nn.ModuleDict()
        self._layers_address_embedding = nn.ParameterDict()
        self._layers_distribution_type_embedding = nn.ParameterDict()
        self._layers_lstm = None
        self._lstm_input_dim = None
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._infer_lstm_state = None
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = distribution_type_embedding_dim
        self._proposal_mixture_components = proposal_mixture_components

    def _init_layers(self):
        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._layers_lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        self._layers_lstm.to(device=util._device)

    def _polymorph(self, batch):
        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for variable in example_trace.variables_controlled:
                address = variable.address
                distribution = variable.distribution

                if address not in self._layers_address_embedding:
                    emb = nn.Parameter(util.to_tensor(torch.zeros(self._address_embedding_dim).normal_()))
                    self._layers_address_embedding[address] = emb

                if distribution.name not in self._layers_distribution_type_embedding:
                    emb = nn.Parameter(util.to_tensor(torch.zeros(self._distribution_type_embedding_dim).normal_()))
                    self._layers_distribution_type_embedding[distribution.name] = emb

                if address not in self._layers_proposal:
                    variable_shape = variable.value.shape
                    if isinstance(distribution, Normal):
                        proposal_layer = ProposalNormalNormalMixture(self._lstm_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape, self._sample_embedding_dim, num_layers=1)
                    elif isinstance(distribution, Uniform):
                        proposal_layer = ProposalUniformTruncatedNormalMixture(self._lstm_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape, self._sample_embedding_dim, num_layers=1)
                    elif isinstance(distribution, Poisson):
                        proposal_layer = ProposalPoissonTruncatedNormalMixture(self._lstm_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape, self._sample_embedding_dim, num_layers=1)
                    elif isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategoricalCategorical(self._lstm_dim, distribution.num_categories)
                        sample_embedding_layer = EmbeddingFeedForward(variable.value.shape, self._sample_embedding_dim, input_is_one_hot_index=True, input_one_hot_dim=distribution.num_categories, num_layers=1)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    proposal_layer.to(device=util._device)
                    sample_embedding_layer.to(device=util._device)
                    self._layers_sample_embedding[address] = sample_embedding_layer
                    self._layers_proposal[address] = proposal_layer
                    layers_changed = True
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address), distribution.name))
        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total addresses: {:,}, distribution types: {:,}, parameters: {:,}'.format(len(self._layers_address_embedding), len(self._layers_distribution_type_embedding), num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def _infer_step(self, variable, prev_variable=None, proposal_min_train_iterations=None):
        success = True
        if prev_variable is None:
            # First time step
            prev_sample_embedding = util.to_tensor(torch.zeros(1, self._sample_embedding_dim))
            prev_address_embedding = util.to_tensor(torch.zeros(self._address_embedding_dim))
            prev_distribution_type_embedding = util.to_tensor(torch.zeros(self._distribution_type_embedding_dim))
            h0 = util.to_tensor(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            c0 = util.to_tensor(torch.zeros(self._lstm_depth, 1, self._lstm_dim))
            self._infer_lstm_state = (h0, c0)
        else:
            prev_address = prev_variable.address
            prev_distribution = prev_variable.distribution
            prev_value = prev_variable.value
            if prev_value.dim() == 0:
                prev_value = prev_value.unsqueeze(0)
            if prev_address in self._layers_address_embedding:
                prev_sample_embedding = self._layers_sample_embedding[prev_address](prev_value.float())
                prev_address_embedding = self._layers_address_embedding[prev_address]
                prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution.name]
            else:
                print('Warning: address of previous variable unknown by inference network: {}'.format(prev_address))
                success = False

        current_address = variable.address
        current_distribution = variable.distribution
        if current_address in self._layers_address_embedding:
            current_address_embedding = self._layers_address_embedding[current_address]
            current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution.name]
        else:
            print('Warning: address of current variable unknown by inference network: {}'.format(current_address))
            success = False

        if success:
            t = torch.cat([self._infer_observe_embedding[0],
                           prev_sample_embedding[0],
                           prev_distribution_type_embedding,
                           prev_address_embedding,
                           current_distribution_type_embedding,
                           current_address_embedding]).unsqueeze(0)
            lstm_input = t.unsqueeze(0)
            lstm_output, self._infer_lstm_state = self._layers_lstm(lstm_input, self._infer_lstm_state)
            proposal_input = lstm_output[0]
            proposal_layer = self._layers_proposal[current_address]
            if proposal_min_train_iterations is not None:
                if proposal_layer._total_train_iterations < proposal_min_train_iterations:
                    print(colored('Warning: using prior, proposal not sufficiently trained ({}/{}) for address: {}'.format(proposal_layer._total_train_iterations, proposal_min_train_iterations, current_address), 'yellow', attrs=['bold']))
                    return current_distribution
            proposal_distribution = proposal_layer.forward(proposal_input, [variable])
            return proposal_distribution
        else:
            print(colored('Warning: using prior as proposal for address: {}'.format(current_address), 'yellow', attrs=['bold']))
            return current_distribution

    def _loss(self, batch):
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            observe_embedding = self._embed_observe(sub_batch)
            sub_batch_length = len(sub_batch)
            sub_batch_loss = 0.
            # print('sub_batch_length', sub_batch_length, 'example_trace_length_controlled', example_trace.length_controlled, '  ')

            # Construct LSTM input sequence for the whole trace length of sub_batch
            lstm_input = []
            for time_step in range(example_trace.length_controlled):
                current_variable = example_trace.variables_controlled[time_step]
                current_address = current_variable.address
                if current_address not in self._layers_address_embedding:
                    print(colored('Address unknown by inference network: {}'.format(current_address), 'red', attrs=['bold']))
                    return False, 0
                current_distribution = current_variable.distribution
                current_address_embedding = self._layers_address_embedding[current_address]
                current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution.name]

                if time_step == 0:
                    prev_sample_embedding = util.to_tensor(torch.zeros(sub_batch_length, self._sample_embedding_dim))
                    prev_address_embedding = util.to_tensor(torch.zeros(self._address_embedding_dim))
                    prev_distribution_type_embedding = util.to_tensor(torch.zeros(self._distribution_type_embedding_dim))
                else:
                    prev_variable = example_trace.variables_controlled[time_step - 1]
                    prev_address = prev_variable.address
                    if prev_address not in self._layers_address_embedding:
                        print(colored('Address unknown by inference network: {}'.format(prev_address), 'red', attrs=['bold']))
                        return False, 0
                    prev_distribution = prev_variable.distribution
                    smp = util.to_tensor(torch.stack([trace.variables_controlled[time_step - 1].value.float() for trace in sub_batch]))
                    prev_sample_embedding = self._layers_sample_embedding[prev_address](smp)
                    prev_address_embedding = self._layers_address_embedding[prev_address]
                    prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution.name]

                lstm_input_time_step = []
                for b in range(sub_batch_length):
                    t = torch.cat([observe_embedding[b],
                                   prev_sample_embedding[b],
                                   prev_distribution_type_embedding,
                                   prev_address_embedding,
                                   current_distribution_type_embedding,
                                   current_address_embedding])
                    lstm_input_time_step.append(t)
                lstm_input.append(torch.stack(lstm_input_time_step))

            # Execute LSTM in a single operation on the whole input sequence
            lstm_input = torch.stack(lstm_input)
            h0 = util.to_tensor(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            c0 = util.to_tensor(torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim))
            lstm_output, _ = self._layers_lstm(lstm_input, (h0, c0))

            # Construct proposals for each time step in the LSTM output sequence of sub_batch
            for time_step in range(example_trace.length_controlled):
                variable = example_trace.variables_controlled[time_step]
                address = variable.address
                proposal_input = lstm_output[time_step]
                variables = [trace.variables_controlled[time_step] for trace in sub_batch]
                values = torch.stack([v.value for v in variables])
                proposal_layer = self._layers_proposal[address]
                proposal_layer._total_train_iterations += 1
                proposal_distribution = proposal_layer.forward(proposal_input, variables)
                # log_importance_weights = util.to_tensor([trace.log_importance_weight for trace in sub_batch], dtype=torch.float64)
                # importance_weights = torch.exp(log_importance_weights)
                log_prob = proposal_distribution.log_prob(values)
                # print('loss                  ', log_prob)
                # print('log_importance_weights', log_importance_weights)
                # print('importance_weights    ', importance_weights)
                # print()
                if util.has_nan_or_inf(log_prob):
                    print(colored('Warning: NaN, -Inf, or Inf encountered in proposal log_prob.', 'red', attrs=['bold']))
                    print('proposal_distribution', proposal_distribution)
                    print('values', values)
                    print('log_prob', log_prob)
                    print('Fixing -Inf')
                    log_prob = util.replace_negative_inf(log_prob)
                    print('log_prob', log_prob)
                    if util.has_nan_or_inf(log_prob):
                        print(colored('Nan or Inf present in proposal log_prob.', 'red', attrs=['bold']))
                        return False, 0
                sub_batch_loss += -torch.sum(log_prob)
            batch_loss += sub_batch_loss
        return True, batch_loss / batch.size
