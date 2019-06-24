import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = util.to_tensor(loc)
        self.scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal',
                         torch_dist=torch.distributions.Normal(self.loc, self.scale))

    def get_input_parameters(self):
        return {'loc': self.loc, 'scale': self.scale}

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.loc, self.scale)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    def to(self, device):
        print(device)
        self.loc.to(device=device)
        self.scale.to(device=device)
        return self
