import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

from .. import util


class LinearBlock(nn.Module):

    def __init__(self, input_shape, output_shape):
        super().__init__()
        modules = []

        if input_shape == output_shape:
            linear_dim_rate = 1
        else:
            linear_dim_rate = 1.3 if input_shape < output_shape else 0.7

        def shape_at_layer(state):
            dim = int(input_shape*(linear_dim_rate**state))
            if output_shape > input_shape:
                return min(dim, output_shape)
            else:
                return max(dim, output_shape)

        old_shape = input_shape
        for i in range(2):
            next_shape = shape_at_layer(i+1)
            modules.append((f"dense_{i}", nn.Linear(old_shape, next_shape, bias=False)))
            modules.append((f"batchnorm_{i}", nn.BatchNorm1d(next_shape)))
            modules.append((f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            old_shape = next_shape

        modules.append((f"dense_{i+1}", nn.Linear(old_shape, output_shape)))

        self._lin_block = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        return self._lin_block(x)


class ParameterFromRNN(nn.Module):

    r"""

    """
    def __init__(self, input_shape, H, W):
        super().__init__()
        self._input_shape = input_shape
        self._lstm_hidden_shape = 256

        self._iterative_dim = int(H>W) # We want to iterate along the biggest dimension
        self._output_shape = W if W>H else H
        self._seq_len = H if W>H else W

        self._rnn_styles = LinearBlock(input_shape, self._output_shape)

        self.c0 = nn.Parameter(torch.zeros([1, 1, self._lstm_hidden_shape]), requires_grad=False)
        self.h0 = nn.Parameter(torch.zeros([1, 1, self._lstm_hidden_shape]), requires_grad=False)
        self._lstm = nn.LSTM(self._output_shape*2, self._lstm_hidden_shape)

        self._output_blocks = nn.ModuleList([
            LinearBlock(self._lstm_hidden_shape, self._output_shape) for _ in range(self._seq_len)])

    def forward(self, x):
        batch_size = x.size(0)
        x_input = self._rnn_styles(x)
        output = []
        x = torch.cat([x_input, torch.zeros_like(x_input)], dim=1)
        for i, output_block in enumerate(self._output_blocks):
            if i == 0:
                out, hidden_state = self._lstm(x.view(1, batch_size, -1))
            else:
                out, hidden_state = self._lstm(x.view(1, batch_size, -1), hidden_state)
            out = output_block(out.squeeze(0))
            output.append(out) # squeeze sequence
            x = torch.cat([x_input.view(out.shape), out], dim=1) # seq_len x batch_size x *
        output = torch.stack(output, dim=self._iterative_dim)
        return output
