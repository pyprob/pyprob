import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

from .. import util

class LinearBlock(nn.Module):

    def __init__(self, linear_dim, output_dim):
        super().__init__()

        modules = []

        for i in range(8):
            if i == 0:
                modules.append((f"dense_{i}", nn.Linear(linear_dim, output_dim)))
            else:
                modules.append((f"dense_{i}", nn.Linear(output_dim, output_dim)))
            modules.append((f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        self._lin_block = nn.Sequential(OrderedDict(modules))
        self._output_dim = output_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = self._lin_block(x)
        return x

class NormAdaIn(nn.Module):

    def __init__(self, latent_dim, channels):
        """Given a latent vector we seek to map that into "styles" which should be a
        scaling factor and translation factor for each channel --> therefore we
        construct a linear linear layer with output equal twice the number of channels

        """
        super().__init__()
        self.lin = nn.Linear(latent_dim, channels*2)
        self._channels = channels


    def forward(self, x, latents):
        styles = self.lin(latents)
        batch_size = x.size(0)

        # x.dim() - 2 as we need to extend styles to match the remaining number
        # of dimensions of x (we build styles to match the first two dimensions
        # by default)
        shape = torch.Size([batch_size, self._channels] + (x.dim()-2)*[1])
        scale = styles[:,:self._channels].view(shape)
        bias = styles[:,self._channels:].view(shape)

        x = nn.InstanceNorm2d(self._channels)(x) # see - https://pytorch.org/docs/stable/nn.html?highlight=instancenorm#torch.nn.InstanceNorm2d

        return scale*x + bias

class InputBlock(nn.Module):

    def __init__(self, channels, latent_dim, h_init, w_init):
        super().__init__()

        self.const = nn.Parameter(torch.ones(channels, h_init, w_init)).to(device=util._device)
        self.bias = nn.Parameter(torch.ones(channels)).view(channels,1,1).to(device=util._device)

        self.ada1 = NormAdaIn(latent_dim, channels)
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=(3,3), padding=0)
        self.ada2 = NormAdaIn(latent_dim, channels)

    def forward(self, latents):
        batch_size = latents.size(0)
        x = self.const.expand(torch.Size([batch_size]) + self.const.shape) + self.bias # broadcast the bias
        x = self.ada1(x, latents)
        x = self.conv2d(x)
        x = self.ada2(x, latents)

        return x

class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, latent_dim, h, w):
        super().__init__()

        self.upsample = nn.Upsample(size=(h+4, w+4), mode='nearest')
        self.conv2d1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=0)
        self.ada1 = NormAdaIn(latent_dim, out_channels)
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=0)
        self.ada2 = NormAdaIn(latent_dim, out_channels)

    def forward(self, x, latents):
        x = self.upsample(x)
        x = self.conv2d1(x)
        x = self.ada1(x, latents)
        x = self.conv2d2(x)
        x = self.ada2(x, latents)

        # TODO MAKE SURE LATENTS HAVENT CHANGED
        return x

class ConvTranspose2d(nn.Module):

    """StyleGan: Two dimensional deconvolution based on upsampling

    Here we implement the network presented in "A Style-Based Generator
    Architecture for Generative Adversarial Networks"

    WE DO NOT ADD NOISE AS WE SEEK A DETERMINISTIC FUNCTION (NOT USED IN A
    GAN-like setting)

    ===================================================================
    CITE:
    KARRAS, Tero; LAINE, Samuli; AILA, Timo. A style-based generator
    architecture for generative adversarial networks. In: Proceedings of the
    IEEE Conference on Computer Vision and Pattern Recognition. 2019. p.
    4401-4410.
    ===================================================================

    Implementation is mainly inspired by:

    - https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb

    """

    def __init__(self, linear_dim, H, W):
        super().__init__()
        self.linear_dim = linear_dim
        self._latent_dim = 512 # as done in the paper
        final_channels = 1 # should be 3 if we have colored images TODO

        max_resolution = max(H,W)
        resolution_log2 = int(np.ceil(np.log2(max_resolution)))

        assert H >= 4 and W >= 4

        fmap_max = 512
        fmap_decay = 1.2
        fmap_base = max_resolution * 4

        def channels_at_stage(stage):
            return max(min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max),1)

        num_upsampling_blocks = resolution_log2 - 2 # -2 as we start from a 4 x 4 image (and not a 1 x 1)

        channels = channels_at_stage(2)

        self._styles = LinearBlock(linear_dim, self._latent_dim)

        h = 4 # as done in the paper
        w = 4 # as done in the paper
        self._input_block = InputBlock(channels, self._latent_dim, h, w)

        upscale_module = []
        in_channels = channels
        for stage in range(num_upsampling_blocks):
            out_channels = channels_at_stage(stage + 2)
            h = h*2 if 2*h < H else H
            w = w*2 if 2*w < W else W
            upscale_module.append(UpscaleBlock(in_channels, out_channels,
                                               self._latent_dim, h, w))

            in_channels = out_channels

        self._upscaling = nn.ModuleList(upscale_module)
        self._to_rgb = nn.Conv2d(in_channels, final_channels,kernel_size=(1,1))

    def forward(self, x):
        latents = self._styles(x)

        # go through the constant block
        img = self._input_block(latents)

        # pass through all upscalings using the latents in the AdaIn modules
        for i, m in enumerate(self._upscaling):
            img = m(img, latents)

        return self._to_rgb(img)

    def visualize_deconv(self, aspect=1):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,5))

        random_input = torch.randn(self.linear_dim).view(1,-1)
        viz = self.forward(random_input).detach().squeeze().numpy()

        m = plt.imshow(viz, aspect=aspect)
        plt.colorbar(m)
        plt.show()
