import torch

from . import Distribution
from .. import util


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            probs = util.to_tensor(probs)
        if logits is not None:
            logits = util.to_tensor(logits)
        torch_dist = torch.distributions.Bernoulli(probs=probs, logits=logits)
        self._probs = torch_dist.probs
        self._logits = torch_dist.logits
        super().__init__(name='Bernoulli', address_suffix='Bernoulli()', torch_dist=torch_dist)

    def __repr__(self):
        return 'Bernoulli(probs:{})'.format(self.probs)

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return self._logits
