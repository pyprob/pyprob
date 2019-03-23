import torch
import torch.nn as nn
from termcolor import colored

from . import InferenceNetwork, ProposalNormalNormalMixture, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, ProposalPoissonTruncatedNormalMixture
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
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for variable in example_trace.variables_controlled:
                address = variable.address
                distribution = variable.distribution
                variable_shape = variable.value.shape
                if address not in self._layers_proposal:
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address), distribution.name))
                    if isinstance(distribution, Normal):
                        layer = ProposalNormalNormalMixture(self._observe_embedding_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                    elif isinstance(distribution, Uniform):
                        layer = ProposalUniformTruncatedNormalMixture(self._observe_embedding_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                    elif isinstance(distribution, Poisson):
                        layer = ProposalPoissonTruncatedNormalMixture(self._observe_embedding_dim, variable_shape, mixture_components=self._proposal_mixture_components)
                    elif isinstance(distribution, Categorical):
                        layer = ProposalCategoricalCategorical(self._observe_embedding_dim, distribution.num_categories)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    layer.to(device=util._device)
                    self._layers_proposal[address] = layer
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
        if address in self._layers_proposal:
            proposal_layer = self._layers_proposal[address]
            if proposal_min_train_iterations is not None:
                if proposal_layer._total_train_iterations < proposal_min_train_iterations:
                    print(colored('Warning: using prior, proposal not sufficiently trained ({}/{}) for address: {}'.format(proposal_layer._total_train_iterations, proposal_min_train_iterations, address), 'yellow', attrs=['bold']))
                    return distribution
            proposal_distribution = proposal_layer.forward(self._infer_observe_embedding, [variable])
            return proposal_distribution
        else:
            print(colored('Warning: using prior, no proposal layer for address: {}'.format(address), 'yellow', attrs=['bold']))
            return distribution

    def _loss(self, batch):
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            observe_embedding = self._embed_observe(sub_batch)
            sub_batch_loss = 0.
            for time_step in range(example_trace.length_controlled):
                address = example_trace.variables_controlled[time_step].address
                if address not in self._layers_proposal:
                    print(colored('Address unknown by inference network: {}'.format(address), 'red', attrs=['bold']))
                    return False, 0
                variables = [trace.variables_controlled[time_step] for trace in sub_batch]
                values = torch.stack([v.value for v in variables])
                proposal_layer = self._layers_proposal[address]
                proposal_layer._total_train_iterations += 1
                proposal_distribution = proposal_layer.forward(observe_embedding, variables)
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
