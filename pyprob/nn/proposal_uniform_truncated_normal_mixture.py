import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import TruncatedNormal, Mixture


class ProposalUniformTruncatedNormalMixture(nn.Module):
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
        means = x[:, :self._mixture_components].view(batch_size, -1)
        stddevs = x[:, self._mixture_components:2*self._mixture_components].view(batch_size, -1)
        coeffs = x[:, 2*self._mixture_components:].view(batch_size, -1)
        means = torch.sigmoid(means)
        stddevs = torch.sigmoid(stddevs)
        coeffs = torch.softmax(coeffs, dim=1)
        means = means.view(batch_size, -1)
        stddevs = stddevs.view(batch_size, -1)
        prior_lows = torch.stack([util.to_tensor(v.distribution.low) for v in prior_variables]).view(batch_size)
        prior_highs = torch.stack([util.to_tensor(v.distribution.high) for v in prior_variables]).view(batch_size)
        prior_range = (prior_highs - prior_lows).view(batch_size, -1)
        means = prior_lows.view(batch_size, -1) + (means * prior_range)
        # stddevs = stddevs * prior_stddevs
        stddevs = (prior_range / 1000) + (stddevs * prior_range * 10)
        distributions = [TruncatedNormal(means[:, i:i+1].view(batch_size), stddevs[:, i:i+1].view(batch_size), low=prior_lows, high=prior_highs) for i in range(self._mixture_components)]
        return Mixture(distributions, coeffs)
