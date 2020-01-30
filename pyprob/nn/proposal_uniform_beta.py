import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Beta


class ProposalUniformBeta(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self.output_dim = util.prod(output_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self.output_dim * 2]),
                                        num_layers=num_layers, activation=torch.relu,
                                        activation_last=None,
                                        hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self._softplus = nn.Softplus()

    def forward(self, x, prior_distribution):
        batch_size = x.size(0)
        x = self._ff(x)
        concentration1s = self._softplus(x[:, :self.output_dim].view(batch_size, -1))
        concentration0s = self._softplus(x[:, self.output_dim:].view(batch_size, -1))
        prior_lows = prior_distribution.low.view(batch_size, -1)
        prior_highs = prior_distribution.high.view(batch_size, -1)
        return Beta(concentration1s, concentration0s, low=prior_lows, high=prior_highs)