import torch
import torch.nn as nn

from .. import util


class EmbeddingCNN3D5C(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = util.to_size(input_shape)  # expecting 4d: [channels, depth, height, width]
        self._output_shape = util.to_size(output_shape)
        input_channels = self._input_shape[0]
        self._output_dim = util.prod(self._output_shape)
        self._conv1 = nn.Conv3d(input_channels, 64, 3)
        self._conv2 = nn.Conv3d(64, 64, 3)
        self._conv3 = nn.Conv3d(64, 128, 3)
        self._conv4 = nn.Conv3d(128, 128, 3)
        self._conv5 = nn.Conv3d(128, 128, 3)
        cnn_output_dim = self._forward_cnn(torch.zeros(self._input_shape).unsqueeze(0)).nelement()
        self._lin1 = nn.Linear(cnn_output_dim, self._output_dim)
        # self._lin2 = nn.Linear(self._output_dim, self._output_dim)
        print(input_shape)
        print(cnn_output_dim)
        print(output_shape)
        
    def _forward_cnn(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = nn.MaxPool3d(2)(x)
        x = torch.relu(self._conv3(x))
        x = torch.relu(self._conv4(x))
        x = torch.relu(self._conv5(x))
        x = nn.MaxPool3d(2)(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(torch.Size([batch_size]) + self._input_shape)
        x = self._forward_cnn(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self._lin1(x))
        # x = torch.relu(self._lin2(x))
        return x.view(torch.Size([-1]) + self._output_shape)
