# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
import torch
import torch.nn as nn

from .. import util


class EmbeddingAlexNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)  # expecting 3d: [channels, height, width]
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        self._conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self._conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self._conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self._conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        self._lin2 = nn.Linear(self._output_dim, self._output_dim)

    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        x = torch.relu(self._lin2(x))
        return x.view(torch.Size([-1]) + self._output_shape)
