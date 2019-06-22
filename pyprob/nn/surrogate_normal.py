import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward, ConvTranspose2d
from .. import util
from ..distributions import Distribution, Normal

class SurrogateNormal(nn.Module):
    def __init__(self, input_shape, mean_shape, var_shape, constants={}, em_type='ff', num_layers=2, hidden_dim= None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._mean_output_dim = util.prod(mean_shape)
        self._var_output_dim = util.prod(var_shape)
        self._mean_shape = mean_shape
        self._var_shape = var_shape
        self.em_type = em_type
        constant_mean = False
        constant_var = False
        self.train = True

        if 'loc' in constants:
            mean = constants['loc']
            constant_mean = True
        if 'scale' in constants:
            stddev = constants['scale']
            constant_var = True
        if constant_var and constant_mean:
            self.train = False
            self.dist_type = Normal(loc=mean, scale=stddev)
        elif constant_var or constant_mean:
            raise NotImplementedError("Only supports both loc and scale being constants or neither!")
        else:
            if self.em_type == 'ff':
                self._em = EmbeddingFeedForward(input_shape=input_shape,
                                                output_shape=torch.Size([self._mean_output_dim
                                                                        + self._var_output_dim]),
                                                num_layers=num_layers,
                                                activation=torch.relu,
                                                hidden_dim=hidden_dim,
                                                activation_last=None)

            elif self.em_type == 'deconv':
                # ONLY SUPPORTS DECONV ON THE MEAN
                if len(var_shape) != 1:
                    if not mean_shape[0]*mean_shape[1] == 1:
                        input_dim = input_shape.numel()
                        self._lin_embed = nn.Sequential(nn.Linear(input_dim,
                                                                2*(input_dim + self._var_shape.numel())),
                                                        nn.LeakyReLU(inline=True),
                                                        nn.Linear(2*(input_dim + self._var_shape.numel()),
                                                                input_dim + self._var_shape.numel()),
                        )
                        W, H = mean_shape[1], mean_shape[0]
                        self._em = ConvTranspose2d(input_shape.numel(), W, H)
                        self._input_dim = input_shape.numel()
                    else:
                        raise ValueError('Cannot deconv to a scalar mean')
                else:
                    raise ValueError('Only support independent variable - variance has to be vector')
            self.dist_type = Normal(loc=torch.zeros(mean_shape), scale=torch.ones(var_shape))


    def forward(self, x):
        if self.train:
            batch_size = x.size(0)
            if self.em_type == 'ff':
                x = self._em(x)
                self.means = x[:, :self._mean_output_dim].view(batch_size, *self._mean_shape)
                self.stddevs = torch.exp(x[:, self._mean_output_dim:]).view(batch_size, *self._var_shape)

            elif self.em_type == 'devonv':
                x = self._lin_embed(x)
                mean_embeds = x[batch_size, :self.input_dim].view(batch_size,
                                                                self._input_dim)
                self.means = self._em(mean_embeds)
                self.stddevs = torch.exp(x[batch_size, self.input_dim:]).view(batch_size,
                                                                        self._var_output_dim)
            return Normal(self.means, self.stddevs)
        else:
            return Normal(self.loc.repeat(batch_size, 1,1).squeeze(),
                          self.scale.repeat(batch_size, 1, 1).squeeze())

    def loss(self, p_normal):
        if self.train:
            q_normal = Normal(self.means, self.stddevs)

            return Distribution.kl_divergence(p_normal, q_normal)
        else:
            batch_size = len(distributions)
            return torch.zeros([batch_size,1])
