import torch
import torch.nn as nn
from termcolor import colored

from . import InferenceNetwork, EmbeddingFeedForward, ProposalNormalNormalMixture, \
    ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, \
    ProposalPoissonTruncatedNormalMixture, PriorDist
from .. import util
from ..distributions import Normal, Uniform, Categorical, Poisson


class InferenceNetworkLSTM(InferenceNetwork):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FEEDFORWARD, 'reshape': [10, 10], 'dim': 32, 'depth': 2}}
    def __init__(self, lstm_dim=512, lstm_depth=1, sample_embedding_dim=4,
                 address_embedding_dim=64, distribution_type_embedding_dim=8,
                 proposal_mixture_components=10, variable_embeddings={}, *args, **kwargs):
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
        self._variable_embeddings = variable_embeddings

    def _init_layers(self):
        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim \
                               + self._sample_attention_embedding_dim \
                               + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)

        self._layers_lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        self._layers_lstm.to(device=util._device)

    def _polymorph(self, batch):
        layers_changed = False
        for meta_data, torch_data in batch.sub_batches:
            for time_step in meta_data['latent_time_steps']:
                address = meta_data['addresses'][time_step]
                distribution_name = meta_data['distribution_names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                current_name = meta_data['names'][time_step]

                # do not create proposal layer for observed variables!
                if (current_name in self._observe_embeddings) or (not current_controlled):
                    continue

                if address not in self._layers_address_embedding:
                    emb = nn.Parameter(torch.zeros(self._address_embedding_dim).normal_().to(device=util._device))
                    self._layers_address_embedding[address] = emb

                if distribution_name not in self._layers_distribution_type_embedding:
                    emb = nn.Parameter(torch.zeros(self._distribution_type_embedding_dim).normal_().to(device=util._device))
                    self._layers_distribution_type_embedding[distribution_name] = emb

                if address not in self._layers_proposal:
                    # get variable shape (ignore batch_size)
                    variable_shape = torch_data[time_step]['values'][0].shape
                    if current_name in self._variable_embeddings:
                        var_embedding = self._variable_embeddings[current_name]
                    else:
                        var_embedding = {'num_layers': 2,
                                         'hidden_dim': None}
                    if not current_controlled:
                        proposal_layer = PriorDist()
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Normal':
                        proposal_layer = ProposalNormalNormalMixture(self._lstm_dim,
                                                                     variable_shape,
                                                                     mixture_components=self._proposal_mixture_components,
                                                                     **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Uniform':
                        proposal_layer = ProposalUniformTruncatedNormalMixture(self._lstm_dim,
                                                                               variable_shape,
                                                                               mixture_components=self._proposal_mixture_components,
                                                                               **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim, num_layers=1)
                    elif distribution_name == 'Poisson':
                        proposal_layer = ProposalPoissonTruncatedNormalMixture(self._lstm_dim,
                                                                               variable_shape,
                                                                               mixture_components=self._proposal_mixture_components,
                                                                               **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape, self._sample_embedding_dim, num_layers=1)
                    elif distribution_name == 'Categorical':
                        num_categories = torch_data[time_step]['distribution'].num_categories
                        proposal_layer = ProposalCategoricalCategorical(self._lstm_dim,
                                                                        num_categories,
                                                                        **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      input_is_one_hot_index=True,
                                                                      input_one_hot_dim=num_categories, num_layers=1)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution_name))
                    proposal_layer.to(device=util._device)
                    sample_embedding_layer.to(device=util._device)
                    self._layers_sample_embedding[address] = sample_embedding_layer
                    self._layers_proposal[address] = proposal_layer

                    if self.prev_sample_attention:
                        if distribution_name == 'Categorical':
                            num_categories = torch_data[time_step]['distribution'].num_categories
                            kwargs = {"input_is_one_hot_index": True,
                                      "input_one_hot_dim": num_categories}
                        else:
                            kwargs = {}
                        super()._polymorph_attention(address, variable_shape, kwargs)

                    layers_changed = True
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address), distribution_name))
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
            prev_sample_embedding = torch.zeros(1, self._sample_embedding_dim).to(device=util._device)
            prev_address_embedding = torch.zeros(1, self._address_embedding_dim).to(device=util._device)
            prev_distribution_type_embedding = torch.zeros(1, self._distribution_type_embedding_dim).to(device=util._device)
            h0 = torch.zeros(self._lstm_depth, 1, self._lstm_dim).to(device=util._device)
            c0 = torch.zeros(self._lstm_depth, 1, self._lstm_dim).to(device=util._device)
            self._infer_lstm_state = (h0, c0)
        else:
            prev_address = prev_variable.address
            prev_distribution = prev_variable.distribution
            prev_value = prev_variable.value.to(device=util._device)
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
        current_distribution_name = variable.distribution_name
        if current_address in self._layers_address_embedding:
            current_address_embedding = self._layers_address_embedding[current_address]
            current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution_name]
            if self.prev_sample_attention:
                prev_sample_embedding_attention = self.prev_samples_embedder(current_address,
                                                                             self._infer_observe_embedding,
                                                                             batch_size=1)
            else:
                prev_sample_embedding_attention = torch.Tensor([[]])
        else:
            print('Warning: address of current variable unknown by inference network: {}'.format(current_address))
            success = False

        if success:
            t = torch.cat([self._infer_observe_embedding,
                           prev_sample_embedding,
                           prev_distribution_type_embedding,
                           prev_address_embedding,
                           current_distribution_type_embedding,
                           current_address_embedding,
                           prev_sample_embedding_attention[0]]).unsqueeze(0)
            lstm_input = t.unsqueeze(0)
            lstm_output, self._infer_lstm_state = self._layers_lstm(lstm_input, self._infer_lstm_state)
            proposal_input = lstm_output[0]
            proposal_layer = self._layers_proposal[current_address]
            if proposal_min_train_iterations is not None:
                if proposal_layer._total_train_iterations < proposal_min_train_iterations:
                    current_distribution = variable.distribution
                    print(colored('Warning: using prior, proposal not sufficiently trained ({}/{}) for address: {}'.format(proposal_layer._total_train_iterations, proposal_min_train_iterations, current_address), 'yellow', attrs=['bold']))
                    return current_distribution

            # if variable is uncontrolled the returned distribution is the prior
            proposal_distribution = proposal_layer.forward(proposal_input, variable.distribution.to(device=util._device))
            return proposal_distribution
        else:
            current_distribution = variable.distribution
            print(colored('Warning: using prior as proposal for address: {}'.format(current_address), 'yellow', attrs=['bold']))
            return current_distribution

    def _loss(self, batch):
        batch_loss = 0.
        for meta_data, torch_data in batch.sub_batches:
            observe_embedding = self._embed_observe(meta_data, torch_data)

            sub_batch_length = torch_data[0]['values'].size(0)
            sub_batch_loss = 0.

            # Construct LSTM input sequence for the whole trace length of sub_batch
            lstm_input = []
            for time_step in meta_data['latent_time_steps']:
                current_address = meta_data['addresses'][time_step]
                current_distribution_name = meta_data['distribution_names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                current_name = meta_data['names'][time_step]

                # do not create proposal layer for observed variables!
                if (current_name in self._observe_embeddings) or (not current_controlled):
                    continue

                if current_address not in self._layers_address_embedding:
                    print(colored('Address unknown by inference network: {}'.format(current_address), 'red', attrs=['bold']))
                    return False, 0

                current_address_embedding = self._layers_address_embedding[current_address]
                current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution_name]

                if time_step == 0:
                    if self.prev_sample_attention:
                        self.prev_samples_embedder.init_for_trace()
                    prev_sample_embedding = torch.zeros(sub_batch_length,
                                                        self._sample_embedding_dim).to(device=util._device)
                    prev_address_embedding = torch.zeros(1, self._address_embedding_dim).to(device=util._device)
                    prev_distribution_type_embedding = torch.zeros(1, self._distribution_type_embedding_dim).to(device=util._device)
                else:
                    prev_address = meta_data['addresses'][time_step-1]
                    prev_distribution_name = meta_data['distribution_names'][time_step-1]
                    if prev_address not in self._layers_address_embedding:
                        print(colored('Address unknown by inference network: {}'.format(prev_address), 'red', attrs=['bold']))
                        return False, 0
                    smp = torch_data[time_step-1]['values'].to(device=util._device)
                    prev_sample_embedding = self._layers_sample_embedding[prev_address](smp)
                    prev_address_embedding = self._layers_address_embedding[prev_address]
                    prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution_name]
                    if self.prev_sample_attention:
                        self.prev_samples_embedder.add_value(prev_address, smp)

                if self.prev_sample_attention:
                    prev_sample_embedding_attention = self.prev_samples_embedder(current_address,
                                                                                 observe_embedding,
                                                                                 batch_size=sub_batch_length)
                else:
                    prev_sample_embedding_attention = torch.Tensor([[]]).expand([sub_batch_length,
                                                                                 0]).to(device=util._device)
                # concat to size batch_size x *
                t = torch.cat([observe_embedding,
                               prev_sample_embedding,
                               prev_distribution_type_embedding.repeat(sub_batch_length, 1),
                               prev_address_embedding.repeat(sub_batch_length, 1),
                               current_distribution_type_embedding.repeat(sub_batch_length, 1),
                               current_address_embedding.repeat(sub_batch_length, 1),
                               prev_sample_embedding_attention], dim=1)
                lstm_input.append(t)

            # Execute LSTM in a single operation on the whole input sequence
            lstm_input = torch.stack(lstm_input, dim=0) # dim = seq_len x batch_size x *

            h0 = torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim).to(device=util._device)
            c0 = torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim).to(device=util._device)
            lstm_output, _ = self._layers_lstm(lstm_input, (h0, c0))

            # Construct proposals for each time step in the LSTM output sequence of sub_batch
            for time_step in meta_data['latent_time_steps']:
                current_address = meta_data['addresses'][time_step]
                current_name = meta_data['names'][time_step]

                # do not create proposal layer for observed variables!
                if (current_name in self._observe_embeddings) or (not current_controlled):
                    continue
                else:
                    # only when the variable is controlled do we have a loss propagating through the distribution
                    proposal_input = lstm_output[time_step]
                    values = torch_data[time_step]['values'].to(device=util._device)
                    proposal_layer = self._layers_proposal[current_address]
                    proposal_layer._total_train_iterations += 1
                    proposal_distribution = proposal_layer(proposal_input,
                                                           torch_data[time_step]['distribution'].to(device=util._device))
                    log_prob = proposal_distribution.log_prob(values)

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
