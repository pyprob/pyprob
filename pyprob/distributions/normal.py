import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = util.to_tensor(loc)
        self.scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal',
                         torch_dist=torch.distributions.Normal(self.loc, self.scale))
        self.loc_shape = self.loc.shape
        self.scale_shape = self.scale.shape
        self.dist_argument = {'loc': self.loc.tolist(), 'scale': self.scale.tolist()}

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.loc, self.scale)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
