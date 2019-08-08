import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Uniform

class SurrogateUniform(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, output_shape, constants={}, num_layers=2, hidden_dim=None):
        """
        Surrogate uniform

        !!! CURRENTLY ONLY SUPPORTS COPYING THE SIMULATOR DISTRIBUTION !!!
        """
        super().__init__()
        # input_shape = util.to_size(input_shape)
        # self._output_dim = util.prod(output_shape)
        # self._output_shape = torch.Size([-1]) + output_shape
        # self._ff = EmbeddingFeedForward(input_shape=input_shape,
        #                                 output_shape=torch.Size([self._output_dim * 2]), num_layers=num_layers,
        #                                 activation=torch.relu, activation_last=None)
        # self._total_train_iterations = 0

        if ('low' not in constants) or ('high' not in constants):
            raise NotImplementedError("Uniform distibutions mush CURRENTLY have constants range")
        else:
            self.dist_type = Uniform(low=[constants['low']],
                                    high=[constants['high']])
            self.low = util.to_tensor(constants['low'])
            self.high = util.to_tensor(constants['high'])

    def forward(self, x):
        batch_size = x.size(0)
        # x = self._ff(x)
        # self.low = x[:, :self._output_dim].view(self._output_shape)
        # self.high = torch.exp(x[:, self._output_dim:]).view(self._output_shape)

        return Uniform(low=self.low.repeat(batch_size, 1),
                       high=self.high.repeat(batch_size, 1))

    def _loss(self, p_uniform):
        # simulator_lows = self._transform_low(distributions)
        # simulator_highs = self._transform_high(distributions)
        # p_normal = Uniform(simulator_lows, simulator_highs)
        # q_normal = Uniform(self.low, self.high)

        batch_size = p_uniform.low.size(0)
        return torch.zeros([batch_size,1])
