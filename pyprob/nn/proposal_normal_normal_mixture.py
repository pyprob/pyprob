import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal, Mixture


class ProposalNormalNormalMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, hidden_dim=None, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([3 * self._mixture_components]),
                                        num_layers=num_layers, activation=torch.relu, hidden_dim=hidden_dim,
                                        activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_distribution):
        """ Proposal forward function

        !! The parameters in prior_distribution are required to be batched !!

        """
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2*self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2*self._mixture_components:].view(batch_size, -1)
        stddevs = torch.exp(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_means = prior_distribution.loc.view(batch_size,-1)
        prior_stddevs = prior_distribution.scale.view(batch_size,-1)
        prior_means = prior_means.repeat(1, means.size(-1))
        prior_stddevs = prior_stddevs.repeat(1, stddevs.size(-1))
        print(prior_means, prior_stddevs, means, stddevs)
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        distributions = [Normal(means[:, i:i+1].view(batch_size), stddevs[:, i:i+1].view(batch_size)) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)
