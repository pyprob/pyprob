import torch
import torch.nn as nn
from termcolor import colored

from . import InferenceNetwork, ProposalNormalNormalMixture, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, ProposalPoissonTruncatedNormalMixture, PriorDist
from .. import util
from ..distributions import Normal, Uniform, Categorical, Poisson


class InferenceNetworkFeedForward(InferenceNetwork):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FEEDFORWARD, 'reshape': [10, 10], 'dim': 32, 'depth': 2}}
    def __init__(self, proposal_mixture_components=10, *args, **kwargs):
        super().__init__(network_type='InferenceNetworkFeedForward', *args, **kwargs)
        self._layers_proposal = nn.ModuleDict()
        self._proposal_mixture_components = proposal_mixture_components

    def _init_layers(self):
        pass

    def _polymorph(self, batch):
        layers_changed = False
        for meta_data, torch_data in batch.sub_batches:
            for time_step in meta_data['latent_time_steps']:
                address = meta_data['addresses'][time_step]
                distribution_name = meta_data['distribution_names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                current_name = meta_data['names'][time_step]
                variable_shape = torch_data[time_step]['values'][0].shape
                # do not create proposal layer for observed variables!
                if (current_name in self._observe_embeddings) or (not current_controlled):
                    continue

                if address not in self._layers_proposal:
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address),
                                                                             distribution_name))
                    if not current_controlled:
                        proposal_layer = PriorDist()
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    if distribution_name == 'Normal':
                        layer = ProposalNormalNormalMixture(self._observe_embedding_dim+self._sample_attention_embedding_dim,
                                                            variable_shape)
                    elif distribution_name == 'Uniform':
                        layer = ProposalUniformTruncatedNormalMixture(self._observe_embedding_dim+self._sample_attention_embedding_dim,
                                                                      variable_shape)
                    elif distribution_name == 'Poisson':
                        layer = ProposalPoissonTruncatedNormalMixture(self._observe_embedding_dim+self._sample_attention_embedding_dim,
                                                                      variable_shape)
                    elif distribution_name == 'Categorical':
                        num_categories = torch_data[time_step]['distribution'].num_categories
                        layer = ProposalCategoricalCategorical(self._observe_embedding_dim+self._sample_attention_embedding_dim,
                                                               num_categories)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    layer.to(device=util._device)
                    self._layers_proposal[address] = layer

                    if self.prev_sample_attention:
                        if distribution_name == 'Categorical':
                            num_categories = torch_data[time_step]['distribution'].num_categories
                            kwargs = {"input_is_one_hot_index": True,
                                      "input_one_hot_dim": num_categories}
                        else:
                            kwargs = {}
                        super()._polymorph_attention(address, variable_shape, kwargs)

                    layers_changed = True
        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total addresses: {:,}, parameters: {:,}'.format(len(self._layers_proposal), num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def _infer_step(self, variable, prev_variable=None, proposal_min_train_iterations=None):
        address = variable.address
        distribution = variable.distribution
        if self.prev_sample_attention:
            if prev_variable is None:
                self.prev_samples_embedder.init_for_trace()
            else:
                self.prev_samples_embedder.add_value(prev_variable.address, prev_variable.value)
        if address in self._layers_proposal:
            proposal_layer = self._layers_proposal[address]
            if proposal_min_train_iterations is not None:
                if proposal_layer._total_train_iterations < proposal_min_train_iterations:
                    print(colored('Warning: using prior, proposal not sufficiently trained ({}/{}) for address: {}'.format(proposal_layer._total_train_iterations, proposal_min_train_iterations, address), 'yellow', attrs=['bold']))
                    return distribution
            if self.prev_sample_attention:
                prev_samples_embedding = self.prev_samples_embedder(address,
                                                                    self._infer_observe_embedding, batch_size=1)
                proposal_layer_input = torch.cat([prev_samples_embedding,
                                                  self._infer_observe_embedding], dim=1)
            else:
                proposal_layer_input = self._infer_observe_embedding
            proposal_distribution = proposal_layer.forward(proposal_layer_input,
                                                           variable.distribution.to(device=util._device))
            return proposal_distribution
        else:
            print(colored('Warning: using prior, no proposal layer for address: {}'.format(address), 'yellow', attrs=['bold']))
            return distribution

    def _loss(self, batch):
        batch_loss = 0.
        for meta_data, torch_data in batch.sub_batches:
            observe_embedding = self._embed_observe(meta_data, torch_data)
            sub_batch_length = torch_data[0]['values'].size(0)
            sub_batch_loss = 0.
            prev_time_step = None
            for time_step in meta_data['latent_time_steps']:
                address = meta_data['addresses'][time_step]
                current_controlled = meta_data['controls'][time_step]
                current_name = meta_data['names'][time_step]

                if (current_name in self._observe_embeddings) or (not current_controlled):
                    continue

                if self.prev_sample_attention:
                    if time_step == 0:
                        self.prev_samples_embedder.init_for_trace()
                    else:
                        prev_address = meta_data['addresses'][prev_time_step]
                        smp = torch_data[prev_time_step]['values'].to(device=util._device)
                        self.prev_samples_embedder.add_value(prev_address, smp)
                if address not in self._layers_proposal:
                    print(colored('Address unknown by inference network: {}'.format(address), 'red', attrs=['bold']))
                    return False, 0
                values = torch_data[time_step]['values'].to(device=util._device)
                proposal_layer = self._layers_proposal[address]
                proposal_layer._total_train_iterations += 1
                if self.prev_sample_attention:
                    prev_samples_embedding = self.prev_samples_embedder(address,
                                                                        observe_embedding,
                                                                        batch_size=sub_batch_length)
                    proposal_input = torch.cat([prev_samples_embedding, observe_embedding], dim=1)
                else:
                    proposal_input = observe_embedding
                proposal_distribution = proposal_layer.forward(proposal_input, torch_data[time_step]['distribution'].to(device=util._device))
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
                prev_time_step = time_step
            batch_loss += sub_batch_loss
        return True, batch_loss / batch.size
