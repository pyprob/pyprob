import torch
import torch.nn as nn

from .. import util
from ..distributions import Beta


class ProposalUniform(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = input_shape
        self._input_dim = util.prod(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        hidden_dim = int((self._input_dim + self._output_dim * 2))
        self._lin1 = nn.Linear(self._input_dim, hidden_dim)
        self._lin2 = nn.Linear(hidden_dim, self._output_dim * 2)

    def forward(self, x, prior_variables):
        x = torch.relu(self._lin1(x.view(-1, self._input_dim)))
        x = torch.relu(self._lin2(x))
        concentration1s = 1. + x[:, :self._output_dim].view(self._output_shape)
        concentration0s = 1. + x[:, self._output_dim:].view(self._output_shape)
        prior_lows = torch.stack([v.distribution.low for v in prior_variables]).view(concentration1s.size())
        prior_highs = torch.stack([v.distribution.high for v in prior_variables]).view(concentration1s.size())
        return Beta(concentration1s, concentration0s, low=prior_lows, high=prior_highs)
