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
        constant_loc = False
        constant_scale = False
        self.train = True

        if 'loc' in constants:
            loc = constants['loc']
            constant_loc = True
        if 'scale' in constants:
            scales = constants['scale']
            constant_scale = True
        if constant_scale and constant_loc:
            self.train = False
            self.dist_type = Normal(loc=loc, scale=scales)
        elif constant_scale or constant_loc:
            raise NotImplementedError("Only supports both loc and scale being constants or neither!")
        else:
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
                if loc_shape.numel() != 1:
                    input_dim = input_shape.numel()
                    self._lin_embed = nn.Sequential(nn.Linear(input_dim,
                                                            2*(input_dim + self._scale_shape.numel())),
                                                    nn.LeakyReLU(inplace=True),
                                                    nn.Linear(2*(input_dim + self._scale_shape.numel()),
                                                            input_dim + self._scale_shape.numel()),
                    )
                    W, H = loc_shape[1], loc_shape[0]
                    self._em = ConvTranspose2d(input_shape.numel(), W, H)
                    self._input_dim = input_shape.numel()
                else:
                    raise ValueError('Cannot deconv to a scalar mean')
            self.dist_type = Normal(loc=torch.zeros(loc_shape), scale=torch.ones(scale_shape))

    def forward(self, x):
        if self.train:
            batch_size = x.size(0)
            if self.em_type == 'ff':
                x = self._em(x)
                self.loc = x[:, :self._loc_output_dim].view(batch_size, *self._loc_shape)
                self.scales = torch.exp(x[:, self._loc_output_dim:]).view(batch_size, *self._scale_shape)

            elif self.em_type == 'deconv':
                x = self._lin_embed(x)
                loc_embeds = x[:, :self._input_dim].view(batch_size, self._input_dim)
                self.loc = self._em(loc_embeds).view(batch_size, *self._loc_shape)
                self.scales = torch.exp(x[:, self._input_dim:]).view(batch_size, *self._scale_shape)
            return Normal(self.loc, self.scales)
        else:
            return Normal(self.loc.repeat(batch_size, 1, 1).squeeze(),
                          self.scale.repeat(batch_size, 1, 1).squeeze())

    def loss(self, p_normal):
        if self.train:
            q_normal = Normal(self.loc, self.scales)

            return Distribution.kl_divergence(p_normal, q_normal)
        else:
            batch_size = p_normal.loc.size(0)
            return torch.zeros([batch_size,1])
