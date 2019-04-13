import torch

from . import Distribution
from .. import util


class LogNormal(Distribution):
    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        super().__init__(name='LogNormal', address_suffix='LogNormal', torch_dist=torch.distributions.LogNormal(loc, scale))

    def __repr__(self):
        return 'LogNormal(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
