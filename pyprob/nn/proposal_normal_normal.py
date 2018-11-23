import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal


class ProposalNormalNormal(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._output_dim].view(batch_size, -1)
        stddevs = torch.exp(x[:, self._output_dim:]).view(batch_size, -1)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(means.size())
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(stddevs.size())
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        means = means.view(self._output_shape)
        stddevs = stddevs.view(self._output_shape)
        return Normal(means, stddevs)
