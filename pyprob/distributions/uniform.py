import torch

from . import Distribution
from .. import util


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = util.to_tensor(low)
        self.high = util.to_tensor(high)
        super().__init__(name='Uniform', address_suffix='Uniform',
                         torch_dist=torch.distributions.Uniform(self.low, self.high))
        self.args = {'low': low.tolist(), 'high': high.tolist()}

    def __repr__(self):
        return 'Uniform(low: {}, high: {})'.format(self.low, self.high)

    @property
    def low(self):
        return self._torch_dist.low

    @property
    def high(self):
        return self._torch_dist.high
