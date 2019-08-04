import torch

from . import Distribution
from .. import util


class Exponential(Distribution):
    def __init__(self, rate):
        rate = util.to_tensor(rate)
        super().__init__(name='Exponential', address_suffix='Exponential', torch_dist=torch.distributions.Exponential(rate))

    def __repr__(self):
        return 'Exponential(rate:{})'.format(self.rate)

    @property
    def rate(self):
        return self._torch_dist.mean
