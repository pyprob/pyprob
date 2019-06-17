import torch

from . import Distribution
from .. import util


class Weibull(Distribution):
    def __init__(self, scale, concentration):
        scale = util.to_tensor(scale)
        concentration = util.to_tensor(concentration)
        super().__init__(name='Weibull', address_suffix='Weibull', torch_dist=torch.distributions.Weibull(scale,concentration))

    def __repr__(self):
        return 'Weibull(scale:{},concentration:{})'.format(self.scale, self.concentration)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    @property
    def scale(self):
        return self._torch_dist.scale

    @property
    def concentration(self):
        return self._torch_dist.concentration

