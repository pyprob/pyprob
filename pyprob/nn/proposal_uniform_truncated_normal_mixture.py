import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import TruncatedNormal, Mixture


class ProposalUniformTruncatedNormalMixture(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, hidden_dim=None,
                 mixture_components=10):
        super().__init__()
        # Currently only supports event_shape=torch.Size([]) for the mixture components
        self._mixture_components = mixture_components
        input_shape = util.to_size(input_shape)
        self.output_dim = util.prod(output_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([(2*self.output_dim + 1) * self._mixture_components]),
                                        num_layers=num_layers,
                                        activation=torch.relu, activation_last=None,
                                        hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self._logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, prior_distribution):
        """ Proposal forward function

        !! The parameters in prior_distribution are required to be batched !!
        """
        batch_size = x.size(0)
        x = self._ff(x)

        slice_size = self.output_dim*self._mixture_components
        means = x[:, :slice_size].view(batch_size, -1)
        stddevs = x[:, slice_size:2*slice_size].view(batch_size, -1)
        coeffs = x[:, 2*slice_size:].view(batch_size, -1)

        means = torch.sigmoid(means)
        stddevs = torch.sigmoid(stddevs)
        log_coeffs = self._logsoftmax(coeffs)

        prior_lows = prior_distribution.low.view(batch_size, -1).repeat(1,slice_size)
        prior_highs = prior_distribution.high.view(batch_size, -1).repeat(1, slice_size)
        prior_range = (prior_highs - prior_lows)

        means = prior_lows + (means * prior_range)
        # stddevs = stddevs * prior_stddevs
        stddevs = (prior_range / 1000) + (stddevs * prior_range * 10)

        distributions = [TruncatedNormal(means[:, i*self.output_dim:(i+1)*self.output_dim].view(batch_size,
                                                                                       self.output_dim),
                                         stddevs[:, i*self.output_dim:(i+1)*self.output_dim].view(batch_size,
                                                                                         self.output_dim),
                                         low=prior_distribution.low,
                                         high=prior_distribution.high)
                         for i in range(self._mixture_components)]

        return Mixture(distributions, logits=log_coeffs)
