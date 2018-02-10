import torch
import torch.distributions
from torch.autograd import Variable
import numpy as np
from scipy.misc import logsumexp
import collections
import math

from . import util


class Distribution(object):
    def __init__(self, name, address_suffix='', torch_dist=None):
        self.name = name
        self.address_suffix = address_suffix
        self.torch_dist = torch_dist

    def sample(self):
        return self.torch_dist.sample()

    def log_prob(self, value):
        return self.torch_dist.log_prob(value)

    def expectation(self, func):
        raise NotImplementedError()

    def mean(self):
        if self.torch_dist is not None:
            return self.torch_dist.mean()
        else:
            raise NotImplementedError()


class Empirical(Distribution):
    def __init__(self, values, log_weights=None, combine_duplicates=False):
        length = len(values)
        if log_weights is None:
            log_weights = Variable(torch.zeros(length)).fill_(-math.log(length)) # assume uniform distribution if no weights are given
        else:
            log_weights = util.to_variable(log_weights)
        weights = torch.exp(log_weights - util.logsumexp(log_weights))
        distribution = collections.defaultdict(float)
        # This can be simplified once PyTorch supports content-based hashing of tensors. See: https://github.com/pytorch/pytorch/issues/2569
        if combine_duplicates:
            for i in range(length):
                found = False
                for key, value in distribution.items():
                    if torch.equal(key, values[i]):
                        # Differentiability warning: values[i] is discarded here. If we need to differentiate through all values, the gradients of values[i] and key should be tied here.
                        distribution[key] = value + weights[i]
                        found = True
                if not found:
                    distribution[values[i]] = weights[i]
        else:
            for i in range(length):
                distribution[values[i]] += weights[i]
        values = list(distribution.keys())
        weights = list(distribution.values())
        self.length = len(values)
        weights = torch.cat(weights)
        self.weights, indices = torch.sort(weights, descending=True)
        self.values = [values[int(i)] for i in indices]
        self.weights_np = self.weights.data.cpu().numpy()
        # self.values_np = torch.stack(self.values).data.cpu().numpy()
        super().__init__('Emprical')

    def __len__(self):
        return self.length

    def sample(self):
        return util.fast_np_random_choice(self.values, self.weights_np)

    def expectation(self, func):
        ret = 0.
        for i in range(self.length):
            ret += func(self.values[i]) * self.weights[i]
        return ret

    def mean(self):
        return self.expectation(lambda x: x)


class Normal(Distribution):
    def __init__(self, prior_mean, prior_std):
        self.prior_mean = util.to_variable(prior_mean)
        self.prior_std = util.to_variable(prior_std)
        super(Normal, self).__init__('Normal', '_Normal', torch.distributions.Normal(self.prior_mean, self.prior_std))
