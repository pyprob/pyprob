import torch

from . import Distribution
from .. import util


class MultivariateNormal(Distribution):
    def __init__(self, loc, scale):
        self.loc = util.to_tensor(loc)
        self.scale = util.to_tensor(scale)
        super().__init__(name='MultivariateNormal', address_suffix='MultivariateNormal', torch_dist=torch.distributions.MultivariateNormal(loc, scale))

    def __repr__(self):
        return 'MultivariateNormal(mean:{}, stddev:{})'.format(self.mean, self.stddev)

    def get_input_parameters(self):
        return {'loc': self.loc, 'scale': self.scale}

    def to(self, device):
        self.loc = self.loc.to(device=device)
        self.scale = self.scale.to(device=device)
        super().__init__(name='MultivariateNormal', address_suffix='MultivariateNormal', torch_dist=torch.distributions.MultivariateNormal(loc, scale))
        return self
