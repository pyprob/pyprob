import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Gamma

class SurrogateGamma(nn.Module):

    def __init__(self, input_shape, shape_shape, rate_shape, constants={}, num_layers=2,
                 hidden_dim=None):
        super().__init__()
        self._shape = None
        self._rate = None
        self.constant_shape = False
        self.constant_rate = False
        self.do_train = True
        self._rate_output_dim = util.prod(shape_shape)
        self._shape_output_dim = util.prod(rate_shape)
        self._shape_shape = shape_shape
        self._rate_shape = rate_shape

        input_shape = util.to_size(input_shape)

        if 'shape' in constants:
            self._shape_const = nn.Parameter(constants['shape'].to(device=util._device),requires_grad=False)
            self.constant_shape= True
        if 'rate' in constants:
            self._rate_const = nn.Parameter(constants['rate'].to(device=util._device), requires_grad=False)
            self.constant_rate = True

        if self.constant_shape and self.constant_rate:
            self.do_train = False
        else:
            if self.constant_shape and not self.constant_rate:
                # only loc needs learned
                self._shape_output_dim = 0
            elif self.constant_rate and not self.constant_shape:
                # only scale need learned
                self._rate_output_dim = 0

            self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                            output_shape=torch.Size([self._shape_output_dim + self._rate_output_dim]),
                                            num_layers=num_layers,
                                            activation=torch.relu,
                                            activation_last=None,
                                            hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self.dist_type = Gamma(torch.zeros(self._shape_shape)+0.5, torch.ones(self._rate_shape)+0.5)


    def forward(self, x, no_batch=False):
        batch_size = x.size(0)

        if self.do_train:
            x = self._ff(x)
            if not self.constant_shape:
                self._shape = torch.exp(x[:, :self._shape_output_dim]).view(batch_size, *self._shape_shape)
            else:
                self._shape = self._shape_const.expand(batch_size, 1)

            if not self.constant_rate:
                self._rate = torch.exp(x[:, self._shape_output_dim:]).view(batch_size, *self._rate_shape)
            else:
                self._rate= self._rate_const.expand(batch_size, 1)

            if no_batch:
                self._shape = self._shape.squeeze(0)
                self._rate = self._rate.squeeze(0)
            return Gamma(self._shape, self._rate)

            if no_batch:
                return Gamma(self._shape_const.expand(*self._shape_shape),
                             self._rate_const.expand(*self._rate_shape))
        else:
            return Gamma(self._shape_const.expand(batch_size, *self._shape_shape),
                         self._rate_const.expand(batch_size, *self._rate_shape))

    def _loss(self, p_normal):
        if self.do_train:
            q_normal = Gamma(self._shape, self._rate)

            return Distribution.kl_divergence(p_normal, q_normal)
        else:
            batch_size = p_normal.shape.size(0) # concentration = shape
            return torch.zeros([batch_size,1]).to(device=util._device)
