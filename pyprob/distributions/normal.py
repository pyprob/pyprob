import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):

        self.loc = util.to_tensor(loc)
        self.scale = util.to_tensor(scale)

        if self.loc.dim() == 0:
            self.loc = self.loc.unsqueeze(0)
        if self.scale.dim() == 0:
            self.scale = self.scale.unsqueeze(0)

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

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value).flatten(start_dim=self.flatten_dim_loc))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s.view(self.loc_shape)
        else:
            raise NotImplementedError()

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
