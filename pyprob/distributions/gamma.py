import torch

from . import Distribution
from .. import util


class Gamma(Distribution):
    def __init__(self, concentration, rate):
        concentration = util.to_tensor(concentration)
        rate = util.to_tensor(rate)
        super().__init__(name='Gamma', address_suffix='Gamma', torch_dist=torch.distributions.Gamma(concentration=concentration, rate=rate))

    def __repr__(self):
        return 'Gamma(concentration={}, rate={})'.format(self.concentration.detach().cpu().numpy().tolist(), self.rate.detach().cpu().numpy().tolist())

    @property
    def concentration(self):
        return self._torch_dist.concentration

    @property
    def rate(self):
        return self._torch_dist.rate

    def to(self, device):
        return Gamma(concentration=self.concentration.to(device), rate=self.rate.to(device))
