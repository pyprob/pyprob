import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Bernoulli


class ProposalBernoulliBernoulli(nn.Module):
    def __init__(self, input_shape, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape, output_shape=torch.Size([1]), num_layers=num_layers, activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

    def forward(self, x, prior_variables):
        batch_size = x.size(0)
        x = self._ff(x)
        probs = torch.sigmoid(x).view(batch_size, -1) + util._epsilon
        return Bernoulli(probs)
