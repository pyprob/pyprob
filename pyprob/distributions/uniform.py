import torch

from . import Distribution
from .. import util


class Uniform(Distribution):
    def __init__(self, low, high):
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        super().__init__(name='Uniform', address_suffix='Uniform',
                         torch_dist=torch.distributions.Uniform(self._low, self._high))

    def get_input_parameters(self):
        constants = {'low': self._low, 'high': self._high}
        return constants

    def __repr__(self):
        return 'Uniform(low: {}, high: {})'.format(self._low, self._high)

    @property
    def low(self):
        return self._torch_dist.low

    @property
    def high(self):
        return self._torch_dist.high

    def to(self, device):
        self._low = self._low.to(device=device)
        self._high = self._high.to(device=device)
        super().__init__(name='Uniform', address_suffix='Uniform',
                         torch_dist=torch.distributions.Uniform(self._low, self._high))
        return self
