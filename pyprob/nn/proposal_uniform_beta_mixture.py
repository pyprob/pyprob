import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Beta, Mixture


class ProposalUniformBetaMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([3 * self._mixture_components]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        concentration1s = x[:, :self._mixture_components].view(batch_size, -1)
        concentration0s = x[:, self._mixture_components:2*self._mixture_components].view(batch_size, -1)
        concentration1s = 1. + torch.relu(concentration1s)
        concentration0s = 1. + torch.relu(concentration0s)
        coeffs = x[:, 2*self._mixture_components:].view(batch_size, -1)
        coeffs = torch.softmax(coeffs, dim=1)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(batch_size)
        distributions = [Beta(concentration1s[:, i:i+1].view(batch_size), concentration0s[:, i:i+1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)
