import torch

from . import Distribution
from .. import util


class Weibull(Distribution):
    def __init__(self, scale,concentration):
        self._scale = util.to_tensor(scale)
        self._concentration = util.to_tensor(concentration)
        super().__init__(name='Weibull', address_suffix='Weibull', torch_dist=torch.distributions.Weibull(self._scale,self._concentration))

    def __repr__(self):
        return 'Weibull(scale:{},concentration:{})'.format(self.scale, self.concentration)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    @property
    def scale(self):
        print(" The scale is: {0} \n and has type: {1}".format(self.scale, type(self.scale)))
        return self._scale
    
    @property
    def concentration(self):
        return self._concentration
    
