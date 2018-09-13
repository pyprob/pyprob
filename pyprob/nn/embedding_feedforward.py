import torch
import torch.nn as nn

from .. import util


class EmbeddingFeedForward(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self._input_shape = input_shape
        self._input_dim = util.prod(input_shape)
        self._output_dim = util.prod(output_shape)
        self._output_shape = torch.Size([-1]) + output_shape
        hidden_dim = int((self._input_dim + self._output_dim) / 2)
        self._lin1 = nn.Linear(self._input_dim, hidden_dim)
        self._lin2 = nn.Linear(hidden_dim, self._output_dim)

    def forward(self, x):
        x = torch.relu(self._lin1(x.view(-1, self._input_dim)))
        x = torch.relu(self._lin2(x))
        return x.view(self._output_shape)
