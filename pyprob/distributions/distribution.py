import torch

from .. import util


class Distribution():
    def __init__(self, name, address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size(), torch_dist=None):
        self.name = name
        self._address_suffix = address_suffix
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._torch_dist = torch_dist

    @property
    def batch_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.batch_shape
        else:
            return self._batch_shape

    @property
    def event_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.event_shape
        else:
            return self._event_shape

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        return torch.exp(self.log_prob(util.to_tensor(value)))

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            return self._torch_dist.variance
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    @staticmethod
    def kl_divergence(distribution_1, distribution_2):
        if distribution_1._torch_dist is None or distribution_2._torch_dist is None:
            raise ValueError('KL divergence is not currently supported for this pair of distributions.')
        return torch.distributions.kl.kl_divergence(distribution_1._torch_dist, distribution_2._torch_dist)
