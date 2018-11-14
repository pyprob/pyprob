import torch
import torch.nn as nn
from termcolor import colored

from . import InferenceNetwork, EmbeddingFeedForward, ProposalNormalNormalMixture, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, ProposalPoissonTruncatedNormalMixture
from .. import util
from ..distributions import Normal, Uniform, Categorical, Poisson


class InferenceNetworkLSTM(InferenceNetwork):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FEEDFORWARD, 'reshape': [10, 10], 'dim': 32, 'depth': 2}}
    def __init__(self, lstm_dim=512, lstm_depth=2, sample_embedding_dim=16, address_embedding_dim=256, distribution_type_embedding_dim=16, *args, **kwargs):
        super().__init__(network_type='InferenceNetworkLSTM', *args, **kwargs)
        self._layers_proposal = nn.ModuleDict()
        self._layers_sample_embedding = nn.ModuleDict()
        self._layers_address_embedding = nn.ParameterDict()
        self._layers_distribution_type_embedding = nn.ParameterDict()
        self._layers_lstm = None
        self._lstm_input_dim = None
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = distribution_type_embedding_dim

    def _init_layers(self):
        self._lstm_input_dim = self._observe_embedding_dim + self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)

    def _infer_step(self, variable, previous_variable=None, proposal_min_train_iterations=None):
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

    def _polymorph(self, batch):
        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for variable in example_trace.variables_controlled:
                address = variable.address
                distribution = variable.distribution

                if address not in self._layers_address_embedding:
                    emb = nn.Parameter(util.tensor(torch.zeros(self._address_embedding_dim).normal_()))
                    self._layers_address_embedding[address] = emb

                if distribution.name not in self._layers_distribution_type_embedding:
                    emb = nn.Parameter(util.tensor(torch.zeros(self._distribution_type_embedding_dim).normal_()))
                    self._layers_distribution_type_embedding[distribution.name] = emb

                variable_shape = variable.value.shape
                if address not in self._layers_proposal:
                    if isinstance(distribution, Normal):
                        proposal_layer = ProposalNormalNormalMixture(self._lstm_dim, variable_shape)
                    elif isinstance(distribution, Uniform):
                        proposal_layer = ProposalUniformTruncatedNormalMixture(self._lstm_dim, variable_shape)
                    elif isinstance(distribution, Poisson):
                        proposal_layer = ProposalPoissonTruncatedNormalMixture(self._lstm_dim, variable_shape)
                    elif isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategoricalCategorical(self._lstm_dim, distribution.num_categories)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    proposal_layer.to(device=util._device)
                    # self._layers_sample_embedding[address] = sample_embedding_layer
                    self._layers_proposal[address] = proposal_layer
                    layers_changed = True
                    print('New layers for address: {}'.format(util.truncate_str(address)))
        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total number of parameters: {:,}'.format(num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def _loss(self, batch):
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            observe_embedding = self._embed_observe(sub_batch)
            sub_batch_loss = 0.
            for time_step in range(example_trace.length_controlled):
                address = example_trace.variables_controlled[time_step].address
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
