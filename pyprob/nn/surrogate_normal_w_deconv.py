import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
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
        output_channel1 = max(int(self._output_dim_mean*0.5),1)
        output_channel2 = max(int(output_channel1*0.5),1)
        output_channel3 = max(int(output_channel2*0.5),1)
        output_channel4 = max(int(output_channel3*0.5),1)
        output_channel5 = max(int(output_channel4*0.5),1)
        output_channel6 = max(int(output_channel5*0.5),1)
        output_channel7 = max(int(output_channel6*0.5),1)
        self._ff_means = nn.Linear(self._output_dim_mean, output_channel1*25) # make a output_channel1 x 4 x 4 (by reshaping)
        self._deconv = nn.Sequential(

            nn.Upsample(size=(10+4,4+4), mode="nearest"),
            nn.Conv2d(output_channel1, output_channel2, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel2, output_channel2, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(size=(20+4,6+4), mode="nearest"),
            nn.Conv2d(output_channel2, output_channel3, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel3, output_channel3, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(size=(40+4, 12+4), mode="nearest"),
            nn.Conv2d(output_channel3, output_channel4, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel4, output_channel4, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(size=(80+4, 20+4),mode="nearest"),
            nn.Conv2d(output_channel4, output_channel5, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel5, output_channel5, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(size=(160+4, 25+4),mode="nearest"),
            nn.Conv2d(output_channel5, output_channel6, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel6, output_channel6, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),

            nn.Upsample(size=(H+4, W+4), mode="nearest"),
            nn.Conv2d(output_channel6, output_channel7, kernel_size=(3,3), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channel7, 1, kernel_size=(3,3), padding=0), # output is 1 x H x W
        )
        self._total_train_iterations = 0

        self.dist_type = Normal(loc=0, scale=1)

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
