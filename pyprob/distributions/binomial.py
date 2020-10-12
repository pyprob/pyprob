import torch

from . import Distribution
from .. import util


class Binomial(Distribution):
    def __init__(self, total_count=1, probs=None, logits=None):
        probs = util.to_tensor(probs)
        logits = util.to_tensor(logits)
        super().__init__(name='Binomial', address_suffix='Binomial', torch_dist=torch.distributions.Binomial(total_count=total_count, probs=probs, logits=logits))

    def __repr__(self):
        return 'Binomial(total_count={}, probs={})'.format(self.total_count, self.probs.detach().cpu().numpy().tolist())

    @property
    def total_count(self):
        return self._torch_dist.total_count

    @property
    def probs(self):
        return self._torch_dist.probs

    @property
    def logits(self):
        return self._torch_dist.logits

    def to(self, device):
        return Binomial(total_count=self.total_count, probs=self.probs.to(device))
