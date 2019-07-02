import torch

from . import Distribution
from .. import util


class Categorical(Distribution):
    def __init__(self, probs=None, logits=None):
        self.use_probs = False
        if probs is not None:
            self.use_probs = True
            probs = util.to_tensor(probs)
            if probs.dim() == 0:
                raise ValueError('probs cannot be a scalar.')
        elif logits is not None:
            logits = util.to_tensor(logits)
            if logits.dim() == 0:
                raise ValueError('logits cannot be a scalar.')

        torch_dist = torch.distributions.Categorical(probs=probs, logits=logits)
        self._probs = torch_dist.probs
        self._logits = torch_dist.logits

        if self.use_probs:
            self._num_categories = self._probs.size(-1)
        else:
            self._num_categories = self._logits.size(-1)

        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(self._num_categories),
                         torch_dist=torch_dist)

    def get_input_parameters(self):
        if self.use_probs:
            return {'probs': self._probs}
        else:
            return {'logits': self._logits}

    def __repr__(self):
        return 'Categorical(num_categories: {}, probs:{})'.format(self.num_categories, self._probs)

    @property
    def num_categories(self):
        return self._num_categories

    @property
    def probs(self):
        return self._probs

    @property
    def logits(self):
        return self._logits

    def to(self, device):
        if self.use_probs:
            self._probs = self._probs.to(device=device)
        else:
            self._logits = self._logits.to(device=device)
        torch_dist = torch.distributions.Categorical(probs=self._probs, logits=self._logits)
        super().__init__(name='Categorical', address_suffix='Categorical(len_probs:{})'.format(self._num_categories),
                         torch_dist=torch_dist)
        return self
