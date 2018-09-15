import torch

from . import Distribution
from .. import util


class Poisson(Distribution):
    def __init__(self, rate):
        rate = util.to_tensor(rate)
        super().__init__(name='Poisson', address_suffix='Poisson', torch_dist=torch.distributions.Poisson(rate))

    def __repr__(self):
        return 'Poisson(rate: {})'.format(self.rate)

    @property
    def rate(self):
        return self._torch_dist.mean
