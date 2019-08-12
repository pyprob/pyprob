import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class ConvTranspose2d(nn.Module):

    def __init__(self, linear_dim, W, H):
        super().__init__()

        self.linear_dim = linear_dim
        rate = 0.4
        final_channel_before_flatten = 3.0
        # reduce the number of channels by 50 % at each deconv until < 10 are left
        # assume linear_dim >> W, H
        n_deconv = int(np.ceil((np.log(final_channel_before_flatten) \
                                - np.log(linear_dim))/np.log(rate)))
        assert n_deconv > 0

        print(H,W, n_deconv)

        n_deconv = int(n_deconv)

        out_channel = max(int(linear_dim*rate),1)
        modules = []
        in_channel = linear_dim
        ratio = H/W # let initial height and width try and respect the ratio (at least somewhat)
        if ratio > 1:
            h = 10
            w = int(np.ceil(h/ratio))
        else:
            w = 10
            h = int(np.ceil(w/ratio))

        self.linear = nn.Linear(linear_dim, out_channel*w*h)
        self._start_w = w
        self._start_h = h
        self._start_channel = out_channel

        in_channel = out_channel
        for n in range(n_deconv - 2):
            out_channel = max(int(in_channel*rate),1)
            modules.append(('upsample'+str(n), nn.Upsample(size=(h+4, w+4), mode='nearest')))
            modules.append((f"conv2d_{n}_0", nn.Conv2d(in_channel, out_channel,
                                                        kernel_size=(3,3), padding=0)))
            modules.append((f"leaky_relu_{n}_0", nn.LeakyReLU(inplace=True)))
            modules.append((f"conv2d_{n}_1", nn.Conv2d(out_channel, out_channel,
                                                        kernel_size=(3,3), padding=0)))
            modules.append((f"leaky_relu_{n}_1", nn.LeakyReLU(inplace=True)))
            h = h*2 if 2*h < H else H
            w = w*2 if 2*w < W else W
            in_channel = out_channel
            print(h,w)

        out_channel = int(in_channel*rate)
        modules.append(('upsample'+str(n+1), nn.Upsample(size=(H+4, W+4), mode='nearest')))
        modules.append((f"conv2d_{n+1}_0", nn.Conv2d(in_channel, out_channel,
                                                     kernel_size=(3,3), padding=0)))
        modules.append((f"leaky_relu_{n+1}_0", nn.LeakyReLU(inplace=True)))
        modules.append((f"conv2d_{n+1}_1", nn.Conv2d(out_channel, 1,
                                                     kernel_size=(3,3), padding=0)))

        self._deconv = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        x = self.linear(x).view(-1, self._start_channel, self._start_h, self._start_w)
        return self._deconv(x)

    def visualize_deconv(self):
        random_input = torch.randn(self.linar_dim).view(1,-1)
        return self.forward(random_input).squeeze().numpy()
