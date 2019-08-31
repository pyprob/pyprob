import torch

from . import Distribution
from .. import util

class Gamma(Distribution):

    def __init__(self, shape, rate):
        self.shape = util.to_tensor(shape)
        self.rate = util.to_tensor(rate)
        self.shape_shape = self.shape.shape
        self.rate_shape = self.rate.shape

        super().__init__(name='Gamma', address_suffix='Gamma',
                       torch_dist=torch.distributions.Gamma(self.shape, self.rate))

    def get_input_parameters(self):
        return {'shape': self.shape, 'rate': self.rate}

    def __repr__(self):
        return 'Gamma(shape:{}, rate:{})'.format(self.shape, self.rate)

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)

    def to(self, device):
        self.shape = self.shape.to(device=device)
        self.rate = self.rate.to(device=device)
        super().__init__(name='Gamma', address_suffix='Gamma',
                         torch_dist=torch.distributions.Gamma(self.shape, self.rate))
        return self
