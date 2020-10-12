import torch

from . import Distribution
from .. import util


class Exponential(Distribution):
    def __init__(self, rate):
        rate = util.to_tensor(rate)
        super().__init__(name='Exponential', address_suffix='Exponential', torch_dist=torch.distributions.Exponential(rate))

    def __repr__(self):
        return 'Exponential({})'.format(self.rate.detach().cpu().numpy().tolist())

    @property
    def rate(self):
        return self._torch_dist.rate

    def to(self, device):
        return Exponential(rate=self.rate.to(device))
