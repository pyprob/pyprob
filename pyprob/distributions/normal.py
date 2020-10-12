import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.Normal(loc, scale))

    def __repr__(self):
        return 'Normal({}, {})'.format(self.mean.detach().cpu().numpy().tolist(), self.stddev.detach().cpu().numpy().tolist())

    @property
    def loc(self):
        return self._torch_dist.loc

    @property
    def scale(self):
        return self._torch_dist.scale

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    def to(self, device):
        return Normal(loc=self.loc.to(device), scale=self.scale.to(device))
