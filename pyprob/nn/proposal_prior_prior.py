import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal


class PriorDist(nn.Module):
    def __init__(self):
        super().__init__()
        self._total_train_iterations = 0

    def forward(self, _, prior_distribution):
        return prior_distribution
