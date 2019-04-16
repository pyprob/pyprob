import torch

from . import Distribution
from .. import util


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        self._concentration = util.to_tensor(concentration)
        self._rate = util.to_tensor(rate)
        super().__init__(name='Gamma', address_suffix='Gamma', torch_dist=torch.distributions.Gamma(concentration, rate))

    def __repr__(self):
        return 'Gamma(concentration:{}, rate:{})'.format(self.concentration, self.rate)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    @property
    def concentration(self):
        return self._concentration

    @property
    def rate(self):
        return self._rate
    
    