import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from . import ConvTranspose2d
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormalConvTranspose2d(nn.Module):
    def __init__(self, input_shape, mean_shape, var_shape, num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        H_input = util.prod(input_shape)
        W_input = 1
        self._output_dim_mean = hidden_dim
        self._var_output_dim = util.prod(var_shape)
        self._mean_output_shape = torch.Size([-1]) + mean_shape
        self._var_output_shape = torch.Size([-1]) + torch.Size([1]) + var_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self._output_dim_mean + self._var_output_dim]), num_layers=num_layers,
                                        hidden_dim=hidden_dim,
                                        activation=torch.relu, activation_last=None)
        H = mean_shape[0]
        W = mean_shape[1]

        self._deconv = ConvTranspose2d(self._output_dim_mean, W, H)

        self._total_train_iterations = 0
        self.dist_type = Normal(loc=torch.zeros(mean_shape), scale=torch.ones(var_shape))

    def _transform_mean(self, dists):
        return torch.stack([d.mean for d in dists])

    def _transform_stddev(self, dists):
        return  torch.stack([d.stddev for d in dists])

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        means = x[:, :self._output_dim_mean].view(batch_size, -1)
        means = self._ff_means(means).view(batch_size, -1, 5, 5)
        self.means = self._deconv(means).squeeze(1) # squeeze the channel

        self.stddevs = torch.exp(x[:, self._output_dim_mean:]).view(self._var_output_shape)

        # if we only have one dimensional parameters, squeeze to make them scalars
        if self.means.shape == torch.Size([1]):
            self.means = self.means.squeeze()
            self.stddevs = self.stddevs.squeeze()

        return Normal(self.means, self.stddevs)

    def loss(self, distributions):
        bs = len(distributions)
        simulator_means = self._transform_mean(distributions)
        simulator_stddevs = self._transform_stddev(distributions)
        p_normal = Normal(simulator_means.view(bs,-1), simulator_stddevs.view(bs,-1))
        q_normal = Normal(self.means.view(bs,-1), self.stddevs.view(bs,-1))

        return Distribution.kl_divergence(p_normal, q_normal)
