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
            raise NotImplementedError("Uniform distibutions must CURRENTLY have constants range")
        else:
            self._low = nn.Parameter(util.to_tensor(constants['low']).to(device=util._device), requires_grad=False)
            self._high = nn.Parameter(util.to_tensor(constants['high']).to(device=util._device), requires_grad=False)
            self.dist_type = Uniform(low=self._low,
                                     high=self._high)
        self.low_shape = self._low.shape
        self.high_shape = self._high.shape

    def forward(self, x, no_batch=False):
        batch_size = x.size(0)
        # x = self._ff(x)
        # self.low = x[:, :self._output_dim].view(self._output_shape)
        # self.high = torch.exp(x[:, self._output_dim:]).view(self._output_shape)

        if no_batch:
            return Uniform(low=self._low, high=self._high)
        else:
            return Uniform(low=self._low.repeat(batch_size, *self.low_shape),
                           high=self._high.repeat(batch_size, *self.high_shape))


    def _loss(self, p_uniform):
        # simulator_lows = self._transform_low(distributions)
        # simulator_highs = self._transform_high(distributions)
        # p_normal = Uniform(simulator_lows, simulator_highs)
        # q_normal = Uniform(self.low, self.high)

        batch_size = p_uniform.low.size(0)
        return torch.zeros([batch_size,1]).to(device=util._device)

