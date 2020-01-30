import torch

from . import Distribution
from .. import util


class Poisson(Distribution):
    def __init__(self, rate):
        self._rate = util.to_tensor(rate)
        super().__init__(name='Poisson', address_suffix='Poisson', torch_dist=torch.distributions.Poisson(self._rate))

    def __repr__(self):
        return 'Poisson(rate: {})'.format(self.rate)

    @property
    def rate(self):
        return self._torch_dist.mean

    def get_input_parameters(self):
        return {'rate': self.rate}

    def to(self, device):
        self._rate = self._rate.to(device=device)
        super().__init__(name='Poisson', address_suffix='Poisson', torch_dist=torch.distributions.Poisson(self._rate))
        return self
