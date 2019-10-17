import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Gamma


class ProposalGammaGamma(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        x = self._ff(x)
        concentration = torch.exp(x[:, :self._output_dim].view(self._output_shape))
        scale = torch.exp(x[:, self._output_dim:]).view(self._output_shape)
        return Gamma(concentration, scale)
