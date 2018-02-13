import torch
from torch.autograd import Variable
import torch.distributions
import numpy as np
from scipy.misc import logsumexp
import collections
import math

from . import util


class Distribution(object):
    def __init__(self, name, address_suffix='', torch_dist=None):
        self.name = name
        self.address_suffix = address_suffix
        self._torch_dist = torch_dist

    def sample(self):
        if self._torch_dist is not None:
            return self._torch_dist.sample()
        else:
            raise NotImplementedError()

    def log_prob(self, value):
        value = util.to_variable(value)
        if self._torch_dist is not None:
            return self._torch_dist.log_prob(value)
        else:
            raise NotImplementedError()

    def expectation(self, func):
        raise NotImplementedError()

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
        if self._torch_dist is not None:
            try:
                return self._torch_dist.stddev
            except AttributeError: # This is because of the changing nature of PyTorch distributions. Should be removed when PyTorch stabilizes.
                return self._torch_dist.std
        else:
            return self.variance.sqrt()


class Empirical(Distribution):
    def __init__(self, values, log_weights=None, combine_duplicates=False):
        length = len(values)
        if log_weights is None:
            log_weights = util.to_variable(torch.zeros(length)).fill_(-math.log(length)) # assume uniform distribution if no weights are given
        else:
            log_weights = util.to_variable(log_weights)
        if isinstance(values, Variable) or torch.is_tensor(values):
            values = util.to_variable(values)
        elif isinstance(values, (list, tuple)):
            if isinstance(values[0], Variable) or torch.is_tensor(values[0]):
                values = util.to_variable(values)
        log_weights = log_weights.view(-1)
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
        self.weights_numpy = self.weights.data.cpu().numpy()
        try: # This can fail in the case values are an iterable collection of non-numeric types (strings, etc.)
            self.values_numpy = torch.stack(self.values).data.cpu().numpy()
        except:
            self.values_numpy = None
        self._mean = None
        self._mean_unweighted = None
        self._variance = None
        self._variance_unweighted = None
        super().__init__('Emprical')

    def __len__(self):
        return self.length

    def __repr__(self):
        try:
            return 'Empirical(length:{}, mean:{}, stddev:{})'.format(self.length, self.mean, self.stddev)
        except RuntimeError:
            return 'Empirical(length:{})'.format(self.length)

    def sample(self):
        return util.fast_np_random_choice(self.values, self.weights_numpy)

    def expectation(self, func):
        ret = 0.
        for i in range(self.length):
            ret += func(self.values[i]) * self.weights[i]
        return ret

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self.expectation(lambda x: x)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            mean = self.mean
            self._variance = self.expectation(lambda x: (x - mean)**2)
        return self._variance

    @property
    def mean_unweighted(self):
        if self._mean_unweighted is None:
            total = 0
            for i in range(self.length):
                total += self.values[i]
            self._mean_unweighted = total / self.length
        return self._mean_unweighted

    @property
    def variance_unweighted(self):
        if self._variance_unweighted is None:
            mean_unweighted = self.mean_unweighted
            total = 0
            for i in range(self.length):
                total += (self.values[i] - mean_unweighted)**2
            self._variance_unweighted = total / self.length
        return self._variance_unweighted

    @property
    def stddev_unweighted(self):
        return self.variance_unweighted.sqrt()


class Normal(Distribution):
    def __init__(self, mean, stddev):
        self._mean = util.to_variable(mean)
        self._stddev = util.to_variable(stddev)
        super().__init__('Normal', '_Normal', torch.distributions.Normal(self._mean, self._stddev))

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{})'.format(self._mean, self._stddev)


# Temporary: this needs to be replaced by torch.distributions.Uniform when the new PyTorch version is released
class Uniform(Distribution):
    def __init__(self, low, high):
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        self._mean = (self._high + self._low) / 2
        self._variance = (self._high - self._low).pow(2) / 12
        super().__init__('Uniform', '_Uniform')

    def __repr__(self):
        return 'Uniform(low: {}, high:{})'.format(self._low, self._high)

    def sample(self):
        shape = self._low.size()
        rand = util.to_variable(torch.zeros(shape).uniform_())
        return self._low + rand * (self._high - self._low)

    def log_prob(self, value):
        value = util.to_variable(value)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.lt(self._high).type_as(self._low)
        return torch.log(lb.mul(ub)) - torch.log(self._high - self._low)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance


class Categorical(Distribution):
    def __init__(self, probs):
        self._probs = util.to_variable(probs)
        self.length = self._probs.nelement()
        super().__init__('Categorical', '_Categorical(size:{})'.format(self.length), torch.distributions.Categorical(probs=self._probs))

    def __repr__(self):
        return 'Categorical(probs:{})'.format(self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value):
        value = util.to_variable(value).view(-1).long()
        return self._torch_dist.log_prob(value)
