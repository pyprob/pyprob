import torch

from . import Distribution
from .. import util


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None):
        probs = util.to_tensor(probs)
        logits = util.to_tensor(logits)
        super().__init__(name='Bernoulli', address_suffix='Bernoulli', torch_dist=torch.distributions.Bernoulli(probs=probs, logits=logits))

    def __repr__(self):
        return 'Bernoulli({})'.format(self._torch_dist.probs.detach().cpu().numpy().tolist())

    @property
    def probs(self):
        return self._torch_dist.probs

    @property
    def logits(self):
        return self._torch_dist.logits

    def to(self, device):
        return Bernoulli(probs=self.probs.to(device))
