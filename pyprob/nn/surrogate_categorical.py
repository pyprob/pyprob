import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Categorical

class SurrogateCategorical(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, num_categories, num_layers=2):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([num_categories]), num_layers=num_layers,
                                        activation=torch.relu, activation_last=None)
        self._total_train_iterations = 0

        self.dist_type = Categorical(probs=torch.Tensor([1]))

    def _transform_probs(self, dists):
        return torch.stack([d.probs for d in dists])

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.probs = torch.softmax(x, dim=1).view(batch_size, -1) + util._epsilon

        return Categorical(self.probs)

    def loss(self, distributions):
        simulator_probs = self._transform_probs(distributions)
        p_normal = Categorical(simulator_probs)
        q_normal = Categorical(self.probs)

        return Distribution.kl_divergence(p_normal, q_normal)
