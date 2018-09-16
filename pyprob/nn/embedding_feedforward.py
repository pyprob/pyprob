import torch
import torch.nn as nn

from .. import util


class EmbeddingFeedForward(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=3, activation=torch.relu, activation_last=torch.relu):
        super().__init__()
        self._input_shape = input_shape
        self._input_dim = util.prod(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        if num_layers < 1:
            raise ValueError('Expecting num_layers >= 1')
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(self._input_dim, self._output_dim))
        else:
            hidden_dim = int((self._input_dim + self._output_dim))
            layers.append(nn.Linear(self._input_dim, hidden_dim))
            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, self._output_dim))
        self._activation = activation
        self._activation_last = activation_last
        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            x = layer(x)
            if i == len(self._layers) - 1:
                x = self._activation_last(x)
            else:
                x = self._activation(x)
        return x.view(self._output_shape)
