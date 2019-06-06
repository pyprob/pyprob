import torch
import torch.nn as nn

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Normal


class PriorDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _, prior_variable):

        return prior_variable[0].distribution
