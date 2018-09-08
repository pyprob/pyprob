import torch

from . import Distribution
from .. import util


class Beta(Distribution):
    def __init__(self, concentration1, concentration0):
        concentration1 = util.to_tensor(concentration1)
        concentration0 = util.to_tensor(concentration0)
        super().__init__(name='Beta', address_suffix='Beta', torch_dist=torch.distributions.Beta(concentration1, concentration0))

    @property
    def concentration1(self):
        return self._torch_dist.concentration1

    @property
    def concentration0(self):
        return self._torch_dist.concentration0
