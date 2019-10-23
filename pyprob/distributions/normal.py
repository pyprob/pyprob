import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        torch_dist = torch.distributions.Normal(loc, scale)
        super().__init__(name='Normal', address_suffix='Normal(batch_shape:{})'.format(list(torch_dist.batch_shape)), torch_dist=torch_dist)

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
