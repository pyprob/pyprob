import torch

from . import Distribution
from .. import util


class Normal(Distribution):
    def __init__(self, loc, scale):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        super().__init__(name='Normal', address_suffix='Normal', torch_dist=torch.distributions.Normal(loc, scale))
