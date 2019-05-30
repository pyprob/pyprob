import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormal(nn.Module):
    def __init__(self, input_shape, mean_shape, var_shape, num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._mean_output_dim = util.prod(mean_shape)
        self._var_output_dim = util.prod(var_shape)
        self._mean_shape = mean_shape
        self._var_shape = var_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self._mean_output_dim
                                                                 + self._var_output_dim]),
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        hidden_dim=hidden_dim,
                                        activation_last=None)
        self._total_train_iterations = 0

        self.dist_type = Normal(loc=torch.zeros(mean_shape), scale=torch.ones(var_shape))

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.means = x[:, :self._mean_output_dim].view(batch_size, self._mean_shape)
        self.stddevs = torch.exp(x[:, self._mean_output_dim:]).view(batch_size, self._var_shape)

        return Normal(self.means, self.stddevs)
