import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward, ConvTranspose2d
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormal(nn.Module):
    def __init__(self, input_shape, loc_shape, scale_shape, constants={},
                 em_type='ff', num_layers=2, hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._loc_output_dim = util.prod(loc_shape)
        self._scale_output_dim = util.prod(scale_shape)
        self._loc_shape = loc_shape
        self._scale_shape = scale_shape
        self.em_type = em_type
        self.constant_loc = False
        self.constant_scale = False
        self.do_train = True
        self._loc = None
        self._scale = None

        # check if either loc or scale are constant
        if 'loc' in constants:
            self._loc = constants['loc']
            self.constant_loc = True
        if 'scale' in constants:
            self._scale = constants['scale']
            self.constant_scale = True

        if self.constant_scale and self.constant_loc:
            self.do_train = False
        else:
            if self.constant_loc and not self.constant_scale:
                # only loc needs learned
                self._loc_output_dim = 0
            elif self.constant_scale and not self.constant_loc:
                # only scale need learned
                self._scale_output_dim = 0
            if self.em_type == 'ff':
                self._em = EmbeddingFeedForward(input_shape=input_shape,
                                                output_shape=torch.Size([self._loc_output_dim
                                                                        + self._scale_output_dim]),
                                                num_layers=num_layers,
                                                activation=torch.relu,
                                                hidden_dim=hidden_dim,
                                                activation_last=None)

            elif self.em_type == 'deconv':
                # ONLY SUPPORTS DECONV ON THE MEAN
                if (loc_shape.numel() != 1) and not (self._loc_output_dim==0):
                    input_dim = input_shape.numel()
                    self._lin_embed = EmbeddingFeedForward(input_shape=input_shape,
                                                output_shape=torch.Size([input_dim + self._scale_output_dim]),
                                                num_layers=num_layers,
                                                activation=torch.relu,
                                                hidden_dim=hidden_dim,
                                                activation_last=None)

                    W, H = loc_shape[1], loc_shape[0]
                    self._em = ConvTranspose2d(input_dim, W, H)
                    self._input_dim = input_dim
                else:
                    raise ValueError('Cannot deconv to a scalar mean, or the mean has not be non-constant')


        self.dist_type = Normal(loc=torch.zeros(loc_shape), scale=torch.ones(scale_shape))

    def forward(self, x):
        if self.do_train:
            batch_size = x.size(0)
            if self.em_type == 'ff':
                x = self._em(x)

                if not self.constant_loc:
                    self._loc = x[:, :self._loc_output_dim].view(batch_size, *self._loc_shape)
                else:
                    self._loc = self._loc.repeat(batch_size, 1, 1) if self._loc is None else self._loc

                if not self.constant_scale:
                    self._scale = torch.exp(x[:, self._loc_output_dim:]).view(batch_size, *self._scale_shape)
                else:
                    self._scale = self._scale.repeat(batch_size, 1, 1) if self._scale is None else self._scale
            elif self.em_type == 'deconv':
                x = self._lin_embed(x)
                loc_embeds = x[:, :self._input_dim].view(batch_size, self._input_dim)
                if not self.constant_loc:
                    self._loc = self._em(loc_embeds).view(batch_size, *self._loc_shape)
                else:
                    self._loc = self._loc.repeat(batch_size, 1, 1) if self._loc is None else self._loc

                if not self.constant_scale:
                    self._scale = torch.exp(x[:, self._input_dim:]).view(batch_size, *self._scale_shape)
                else:
                    self._scale = self._scale.repeat(batch_size, 1, 1) if self._scale is None else self._scale

            return Normal(self._loc, self._scale)
        else:
            return Normal(self._loc.repeat(batch_size, 1, 1),
                          self._scale.repeat(batch_size, 1, 1))

    def _loss(self, p_normal):
        if self.do_train:
            q_normal = Normal(self._loc, self._scale)

            return Distribution.kl_divergence(p_normal, q_normal)
        else:
            batch_size = p_normal.loc.size(0)
            return torch.zeros([batch_size,1])
