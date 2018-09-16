import torch
import torch.nn as nn

from .. import util
from ..distributions import Normal


class ProposalNormal(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = input_shape
        self._input_dim = util.prod(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        hidden_dim = int((self._input_dim + self._output_dim) / 2)
        self._lin1 = nn.Linear(self._input_dim, hidden_dim)
        self._lin2 = nn.Linear(hidden_dim, self._output_dim * 2)

    def forward(self, x, prior_variables):
        x = torch.relu(self._lin1(x.view(-1, self._input_dim)))
        x = self._lin2(x)
        means = x[:, :self._output_dim].view(self._output_shape)
        stddevs = torch.exp(x[:, self._output_dim:]).view(self._output_shape)
        prior_means = torch.stack([v.distribution.mean for v in prior_variables]).view(means.size())
        prior_stddevs = torch.stack([v.distribution.stddev for v in prior_variables]).view(stddevs.size())
        means = prior_means + (means * prior_stddevs)
        stddevs = stddevs * prior_stddevs
        return Normal(means, stddevs)
