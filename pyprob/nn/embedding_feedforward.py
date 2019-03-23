import torch
import torch.nn as nn

from .. import util


class EmbeddingFeedForward(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=3, activation=torch.relu, activation_last=torch.relu, input_is_one_hot_index=False, input_one_hot_dim=None):
        super().__init__()
        self._input_shape = util.to_size(input_shape)
        self._output_shape = util.to_size(output_shape)
        self._input_dim = util.prod(self._input_shape)
        self._output_dim = util.prod(self._output_shape)
        self._input_is_one_hot_index = input_is_one_hot_index
        self._input_one_hot_dim = input_one_hot_dim
        if input_is_one_hot_index:
            if self._input_dim != 1:
                raise ValueError('If input_is_one_hot_index==True, input_dim should be 1 (the index of one-hot value in a vector of length input_one_hot_dim.)')
            self._input_dim = input_one_hot_dim
        if num_layers < 1:
            raise ValueError('Expecting num_layers >= 1')
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(self._input_dim, self._output_dim))
        else:
            hidden_dim = int((self._input_dim + self._output_dim)/2)
            layers.append(nn.Linear(self._input_dim, hidden_dim))
            for i in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, self._output_dim))
        self._activation = activation
        self._activation_last = activation_last
        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        if self._input_is_one_hot_index:
            x = torch.stack([util.one_hot(self._input_one_hot_dim, int(v)) for v in x])
        else:
            x = x.view(-1, self._input_dim).float()
        for i in range(len(self._layers)):
            layer = self._layers[i]
            x = layer(x)
            if i == len(self._layers) - 1:
                if self._activation_last is not None:
                    x = self._activation_last(x)
            else:
                x = self._activation(x)
        return x.view(torch.Size([-1]) + self._output_shape)
