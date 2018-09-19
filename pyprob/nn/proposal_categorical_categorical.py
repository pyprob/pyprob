import torch
import torch.nn as nn

from .. import util
from . import EmbeddingFeedForward
from ..distributions import Categorical


class ProposalCategoricalCategorical(nn.Module):
    def __init__(self, input_shape, num_categories, num_layers=3):
        super().__init__()
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([num_categories]), num_layers=num_layers, activation=torch.relu, activation_last=None)

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon
        return Categorical(probs)
