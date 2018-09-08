import torch

from . import Distribution
from .. import util


class Categorical(Distribution):
    def __init__(self, probs):
        probs = util.to_tensor(probs)
        if probs.dim() == 0:
            raise ValueError('probs cannot be a scalar.')
        self._num_categories = probs.size(-1)
        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(probs.size(-1)), torch_dist=torch.distributions.Categorical(probs=probs))

    @property
    def num_categories(self):
        return self._num_categories
