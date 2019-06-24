import torch
import torch.nn as nn
from collections import OrderedDict

class ConvTranspose2d(nn.Module):

    def __init__(self, linear_dim, W, H):
        super.__init__()

        self.linear_dim = linear_dim
        # reduce the number of channels by 40 % at each deconv until < 10 are left
        # assume linear_dim >> W, H > 10
        n_deconv = int(torch.ceil((torch.log(10.0) - torch.log(linear_dim))/torch.log(0.6)))
        assert n_deconv > 0

        n_deconv = int(n_deconv)

        output_channel1 = max(int(linear_dim_*0.6),1)
        output_channel2 = max(int(output_channel1*0.6),1)
        output_channel3 = max(int(output_channel2*0.6),1)
        output_channel4 = max(int(output_channel3*0.6),1)
        output_channel5 = max(int(output_channel4*0.6),1)
        output_channel6 = max(int(output_channel5*0.6),1)
        output_channel7 = max(int(output_channel6*0.6),1)

        modules = []
        in_channel = linear_dim
        w = 5
        h = 5
        for n in range(n_deconv - 1):
            out_channel = int(in_channel*0.6)
            if n == 0:
                modules.append(('linear', nn.Linear(linear_dim,
                                                    out_channel*w*h)))
            else:
                modules.append(('upsample'+str(n), nn.Upsample(size=(h+4, w+4), mode='nearest')))
                modules.append((f"conv2d_{n}_0", nn.Conv2d(in_channel, out_channel,
                                                           kernel_size=(3,3), padding=0)))
                modules.append((f"leaky_relu_{n}_0", nn.LeakyReLU(inplace=True)))
                modules.append((f"conv2d_{n}_1", nn.Conv2d(out_channel, out_channel,
                                                           kernel_size=(3,3), padding=0)))
                modules.append((f"leaky_relu_{n}_1", nn.LeakyReLU(inplace=True)))
                h *= 2
                w *= 2
            in_channel = out_channel
        out_channel = int(in_channel*0.6)
        modules.append(('upsample'+str(n+1), nn.Upsample(size=(H+4, W+4), mode='nearest')))
        modules.append((f"conv2d_{n+1}_0", nn.Conv2d(in_channel, out_channel,
                                                     kernel_size=(3,3), padding=0)))
        modules.append((f"leaky_relu_{n+1}_0", nn.LeakyReLU(inplace=True)))
        modules.append((f"conv2d_{n+1}_1", nn.Conv2d(out_channel, 1,
                                                     kernel_size=(3,3), padding=0)))

        self._deconv = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self._deconv(x)

    def visualize_deconv(self):
        random_input = torch.randn(self.linar_dim).view(1,-1)
        return self.forward(random_input).squeeze().numpy()
