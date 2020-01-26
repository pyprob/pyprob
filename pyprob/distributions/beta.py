import torch

from . import Distribution
from .. import util


class Beta(Distribution):
    def __init__(self, concentration1, concentration0, low=0, high=1):
        self._concentration1 = util.to_tensor(concentration1)
        self._concentration0 = util.to_tensor(concentration0)
        self.concentration1_shape = self._concentration1.shape
        self.concentration0_shape = self._concentration0.shape
        super().__init__(name='Beta', address_suffix='Beta',
                         torch_dist=torch.distributions.Beta(self._concentration1,
                                                             self._concentration0))
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        self._range = self._high - self._low

    def __repr__(self):
        return 'Beta(concentration1:{}, concentration0:{}, low:{}, high:{})'.format(self._concentration1,
                                                                                    self._concentration0,
                                                                                    self._low,
                                                                                    self._high)

    def get_input_parameters(self):
        return {'concentration1': self._concentration1, 'concentration0': self._concentration0}

    @property
    def concentration1(self):
        return self._torch_dist.concentration1

    @property
    def concentration0(self):
        return self._torch_dist.concentration0

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def sample(self):
        return self._low + (super().sample() * self._range)

    def log_prob(self, value, sum=False):
        lp = super().log_prob((util.to_tensor(value) - self._low) / self._range, sum=False) - torch.log(self._range)
        return torch.sum(lp) if sum else lp

    @property
    def mean(self):
        return self._low + (super().mean * self._range)

    @property
    def variance(self):
        return super().variance * self._range * self._range

    def to(self, device):
        self._concentration0.to(device=device)
        self._concentration1.to(device=device)
        super().__init__(name='Beta', address_suffix='Beta',
                         torch_dist=torch.distributions.Beta(self._concentration1,
                                                             self._concentration0))
        return self
