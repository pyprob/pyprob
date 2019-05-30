import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormal(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, mean_shape, var_shape, num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._mean_output_dim = util.prod(mean_shape)
        self._var_output_dim = util.prod(var_shape)
        self._mean_output_shape = torch.Size([-1]) + mean_shape
        self._var_output_shape = torch.Size([-1]) + var_shape
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([self._mean_output_dim
                                                                 + self._var_output_dim]),
                                        num_layers=num_layers,
                                        activation=torch.relu,
                                        hidden_dim=hidden_dim,
                                        activation_last=None)
        self._total_train_iterations = 0

        self.dist_type = Normal(loc=0, scale=1)

    def _transform_mean(self, dists):
        return torch.stack([d.mean for d in dists])

    def _transform_stddev(self, dists):
        return  torch.stack([d.stddev for d in dists])

    def forward(self, x):
        batch_size = x.size(0)
        x = self._ff(x)
        self.means = x[:, :self._mean_output_dim].view(self._mean_output_shape)
        self.stddevs = torch.exp(x[:, self._mean_output_dim:]).view(self._var_output_shape)

        # if we only have one dimensional parameters, squeeze to make them scalars
        if self.means.shape == torch.Size([1]):
            self.means = self.means.squeeze()
            self.stddevs = self.stddevs.squeeze()

        return Normal(self.means, self.stddevs)

    def loss(self, distributions):
        simulator_means = self._transform_mean(distributions)
        simulator_stddevs = self._transform_stddev(distributions)
        p_normal = Normal(simulator_means, simulator_stddevs)
        q_normal = Normal(self.means, self.stddevs)

        return Distribution.kl_divergence(p_normal, q_normal)

    # def old_loss(self, variable_dists):
    #     simulator_means = self._transform_mean(variable_dists)
    #     simulator_stddevs = self._transform_stddev(variable_dists)
    #     inv_stddevs_sqr = torch.reciprocal(torch.pow(self.stddevs,2))
    #     e_sqr = torch.pow(simulator_means,2) + torch.pow(simulator_stddevs,2)
    #     expected_nlog_norm = 0.5*inv_stddevs_sqr*(e_sqr - 2*simulator_means*self.means
    #                                               + torch.pow(self.means,2))
    #     expected_nlog_norm += torch.log(self.stddevs)
    #     return expected_nlog_norm
