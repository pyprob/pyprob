import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):

        self.loc = util.to_tensor(loc)
        self.scale = util.to_tensor(scale)

        self.loc_shape = self.loc.shape
        self.scale_shape = self.scale.shape

        # handles both batched and non-batched locs and scales
        self.flatten_dim_loc = 0 if len(self.loc_shape) <= 1 else 1
        self.flatten_dim_scale = 0 if len(self.scale_shape) <= 1 else 1

        super().__init__(name='Normal', address_suffix='Normal',
                         torch_dist=torch.distributions.Normal(self.loc.flatten(start_dim=self.flatten_dim_loc),
                                                               self.scale.flatten(start_dim=self.flatten_dim_scale)))

    def get_input_parameters(self):
        return {'loc': self.loc, 'scale': self.scale}

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self.loc, self.scale)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    def to(self, device):
        self.loc = self.loc.to(device=device)
        self.scale = self.scale.to(device=device)
        super().__init__(name='Normal', address_suffix='Normal',
                         torch_dist=torch.distributions.Normal(self.loc.flatten(start_dim=self.flatten_dim_loc),
                                                               self.scale.flatten(start_dim=self.flatten_dim_scale)))
        return self
