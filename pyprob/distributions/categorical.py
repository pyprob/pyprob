import torch

from . import Distribution
from .. import util


class Categorical(Distribution):
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            probs = util.to_tensor(probs)
            if probs.dim() == 0:
                raise ValueError('probs cannot be a scalar.')
        if logits is not None:
            logits = util.to_tensor(logits)
            if logits.dim() == 0:
                raise ValueError('logits cannot be a scalar.')
        torch_dist = torch.distributions.Categorical(probs=probs, logits=logits)
        self._probs = torch_dist.probs
        self._logits = torch_dist.logits
        self._num_categories = self._probs.size(-1)
        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(self._probs.size(-1)), torch_dist=torch_dist)

    def __repr__(self):
        return 'Categorical(num_categories: {}, probs:{})'.format(self.num_categories, self.probs)

    @property
    def num_categories(self):
        return self._num_categories

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return self._logits
