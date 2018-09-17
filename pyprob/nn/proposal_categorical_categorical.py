import torch
import torch.nn as nn

from .. import util
from . import EmbeddingFeedForward
from ..distributions import Categorical


class ProposalCategoricalCategorical(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=3):
        super().__init__()
        self._output_dim = util.prod(output_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([self._output_dim]), num_layers=num_layers, activation=torch.relu, activation_last=lambda x: x)

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon
        return Categorical(probs)
