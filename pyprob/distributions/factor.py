import torch

from . import Distribution
from .. import util


# Pseudo-distribution to implement pyprob.factor
# See "3.2.1 Conditioning with Factors" in van de Meent, J.W., Paige, B., Yang, H. and Wood, F., 2018. An introduction to probabilistic programming. arXiv preprint arXiv:1809.10756.
class Factor(Distribution):
    def __init__(self, log_prob=None, log_prob_func=None):
        if log_prob is None:
            if log_prob_func is None:
                raise RuntimeError('Expecting one of log_prob, log_prob_func')
            else:
                self._log_prob = None
                self._log_prob_func = log_prob_func
        else:
            if log_prob_func is not None:
                raise RuntimeError('Expecting only one of log_prob, log_prob_func')
            self._log_prob = util.to_tensor(log_prob)
            self._log_prob_func = None
        super().__init__(name='Factor', address_suffix='Factor')

    def __repr__(self):
        return 'Factor()'

    def to(self, device):
        if self._log_prob is not None:
            return Factor(log_prob=self.log_prob.to(device))
        else:
            return self

    def log_prob(self, value=None, sum=False):
        if self._log_prob is not None:
            return self._log_prob
        else:
            return self._log_prob_func(value)

    def sample(self):
        return None
