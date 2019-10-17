import torch

from . import Distribution
from .. import util


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        concentration = util.to_tensor(concentration)
        rate = util.to_tensor(rate)
        super().__init__(name='Gamma', address_suffix='Gamma', torch_dist=torch.distributions.Gamma(concentration, rate))

    def __repr__(self):
        return 'Gamma(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
