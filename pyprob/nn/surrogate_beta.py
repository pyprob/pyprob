import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Beta

class SurrogateBeta(nn.Module):

    def __init__(self, input_shape, concentration1_shape, concentration0_shape, constants={}, num_layers=2,
                 hidden_dim=None):
        super().__init__()
        self._concentration1 = None
        self._concentration0 = None
        self.constant_concentration1 = False
        self.constant_concentration0 = False
        self.do_train = True
        self._concentration1_output_dim = util.prod(concentration1_shape)
        self._concentration0_output_dim = util.prod(concentration0_shape)
        self._concentration1_shape = concentration1_shape
        self._concentration0_shape = concentration0_shape

        input_shape = util.to_size(input_shape)

        if 'concentration1' in constants:
            self._concentration1_const = nn.Parameter(constants['concentration1'].to(device=util._device),requires_grad=False)
            self.concentration1_shape= True
        if 'concentration0' in constants:
            self._concentration0_const = nn.Parameter(constants['concentration0'].to(device=util._device), requires_grad=False)
            self.concentration0_rate = True

        if self.constant_concentration1 and self.constant_concentration0:
            self.do_train = False
        else:
            if self.constant_concentration1 and not self.constant_concentration0:
                # only loc needs learned
                self._concentration1_output_dim = 0
            elif self.constant_concentration0 and not self.constant_concentration1:
                # only scale need learned
                self._concentration0_output_dim = 0

            self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                            output_shape=torch.Size([self._concentration1_output_dim + self._concentration0_output_dim]),
                                            num_layers=num_layers,
                                            activation_last=None,
                                            hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self.dist_type = Beta(torch.zeros(self._concentration1_shape)+0.5, torch.ones(self._concentration0_shape)+0.5)


    def forward(self, x, no_batch=False):
        batch_size = x.size(0)

        if self.do_train:
            x = self._ff(x)
            if not self.constant_concentration1:
                self._concentration1 = torch.exp(x[:, :self._concentration1_output_dim]).view(batch_size, *self._concentration1_shape)
            else:
                self._concentration1 = self._concentration1_const.expand(batch_size, 1)

            if not self.constant_concentration0:
                self._concentration0 = torch.exp(x[:, self._concentration0_output_dim:]).view(batch_size, *self._rate_shape)
            else:
                self._concentration0= self._rate_const.expand(batch_size, 1)

            if no_batch:
                self._concentration1 = self._concentration1.squeeze(0)
                self._concentration0 = self._concentration0.squeeze(0)
            return Beta(self._concentration1, self._concentration0)

            if no_batch:
                return Beta(self._concentration1_const.expand(*self._concentration1_shape),
                            self._concentration0_const.expand(*self._concentration0_shape))
        else:
            return Gamma(self._concentration1_const.expand(batch_size, *self._concentration1_shape),
                         self._concentration0_const.expand(batch_size, *self._concentration0_shape))

    def _loss(self, values):
        if self.do_train:
            q_normal = Beta(self._concentration1, self._concentration0)
            return -q_normal.log_prob(values)
            #return Distribution.kl_divergence(p_normal, q_normal)
        else:
            batch_size = values.shape.size(0) # concentration = shape
            return torch.zeros([batch_size,1]).to(device=util._device)
