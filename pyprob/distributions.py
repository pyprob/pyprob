import torch
from torch.autograd import Variable
import torch.distributions
import numpy as np
import collections
import math

from . import util


class Distribution(object):
    def __init__(self, name, address_suffix='', torch_dist=None):
        self.name = name
        self.address_suffix = address_suffix
        self._torch_dist = torch_dist
        self.__mean = None
        self.__variance = None
        self.__stddev = None

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            if s.dim() == 1 and self.length_variates == 1:
                s = s.unsqueeze(1)
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value):
        if self._torch_dist is not None:
            value = util.to_variable(value)
            value = value.view(self.length_batch, self.length_variates)
            lp = self._torch_dist.log_prob(value)
            if lp.dim() == 2 and self.length_variates > 1:
                lp = torch.sum(lp, dim=1)
            if lp.dim() == 1:
                lp = lp.unsqueeze(1)
            return lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        value = util.to_variable(value)
        return torch.exp(self.log_prob(value))

    def expectation(self, func):
        self._expectation(func)

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            try:
                return self._torch_dist.variance
            except AttributeError: # This is because of the changing nature of PyTorch distributions. Should be removed when PyTorch stabilizes.
                return self._torch_dist.std.pow(2)
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

    def expectation(self, func):
        raise NotImplementedError()


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
        weights = torch.exp(log_weights - util.log_sum_exp(log_weights))
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
            try:
                self.values_numpy = np.array(self.values)
            except:
                self.values_numpy = None
        self._mean = None
        self._mean_unweighted = None
        self._variance = None
        self._variance_unweighted = None
        self._min = None
        self._max = None
        self._mean = None
        self._variance = None
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


class Categorical(Distribution):
    def __init__(self, probs):
        self._probs = util.to_variable(probs)
        if self._probs.dim() == 1:
            self.length_variates = 1
            self.length_batch = 1
            self.length_categories = self._probs.size(0)
            self._probs = self._probs.unsqueeze(0)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        elif self._probs.dim() == 2:
            self.length_variates = 1
            self.length_batch = self._probs.size(0)
            self.length_categories = self._probs.size(1)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        else:
            raise ValueError('Expecting 1d or 2d (batched) probabilities.')
        super().__init__('Categorical', '_Categorical(length_categories:{})'.format(self.length_categories), torch.distributions.Categorical(probs=self._probs))

    def __repr__(self):
        return 'Categorical(probs:{}, length_variates:{}, length_batch:{})'.format(self._probs, self.length_variates, self.length_batch)

    def __len__(self):
        return self.length_variates

    def sample(self):
        return self._torch_dist.sample()

    def log_prob(self, value):
        value = util.to_variable(value)
        value = util.to_variable(value).view(-1).long()
        ret = self._torch_dist.log_prob(value).unsqueeze(1)
        return ret


class Mixture(Distribution):
    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length_distributions = len(distributions)
        if probs is None:
            self._probs = util.to_variable(torch.zeros(self.length).fill_(1/self.length))
        else:
            self._probs = util.to_variable(probs)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)

        self.length_variates = 1
        if self._probs.dim() == 1:
            self.length_batch = 1
        elif self._probs.dim() == 2:
            self.length_batch = self._probs.size(0)
        else:
            raise ValueError('Expecting 1d or 2d (batched) probabilities.')
        self._mixing_dist = Categorical(self._probs)
        self._mean = None
        self._variance = None
        super().__init__('Mixture', '_Mixture({})'.format(', '.join([d.address_suffix for d in self._distributions])))

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value):
        value = util.to_variable(value).view(self.length_batch, 1)
        return util.log_sum_exp(torch.log(self._probs) + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).t())

    def sample(self):
        if self.length_batch == 1:
            i = int(self._mixing_dist.sample()[0])
            return self._distributions[i].sample()
        else:
            indices = self._mixing_dist.sample()
            ret = []
            for b in range(self.length_batch):
                i = int(indices[b])
                ret.append(self._distributions[i].sample()[b])
            return util.to_variable(ret)

    @property
    def mean(self):
        if self._mean is None:
            means = util.to_variable([d.mean for d in self._distributions]).squeeze(-1)
            if self.length_batch == 1:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
            self._mean = self._mean.unsqueeze(1)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = util.to_variable([(d.mean.squeeze(-1) - self.mean.squeeze(-1)).pow(2) + d.variance.squeeze(-1) for d in self._distributions])
            if self.length_batch == 1:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
            self._variance = self._variance.unsqueeze(1)
        return self._variance


class Normal(Distribution):
    def __init__(self, mean, stddev):
        self._mean = util.to_variable(mean)
        self._stddev = util.to_variable(stddev)
        if self._mean.dim() == 1:
            self.length_variates = self._mean.size(0)
            self.length_batch = 1
        elif self._mean.dim() == 2:
            self.length_variates = self._mean.size(1)
            self.length_batch = self._mean.size(0)
        else:
            print(self._mean.size())
            print(self._stddev.size())
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        super().__init__('Normal', '_Normal', torch.distributions.Normal(self._mean, self._stddev))

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{}, length_variates:{}, length_batch:{})'.format(self._mean, self._stddev, self.length_variates, self.length_batch)

    # Won't be needed when the new PyTorch version is released
    def cdf(self, value):
        value = util.to_variable(value)
        return 0.5 * (1 + torch.erf((value - self._mean) * self._stddev.reciprocal() / math.sqrt(2)))

    # Won't be needed when the new PyTorch version is released
    def icdf(self, value):
        value = util.to_variable(value)
        return self._mean + self._stddev * torch.erfinv(2 * value - 1) * math.sqrt(2)


class TruncatedNormal(Distribution):
    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high):
        self._mean_non_truncated = util.to_variable(mean_non_truncated)
        self._stddev_non_truncated = util.to_variable(stddev_non_truncated)
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        if self._mean_non_truncated.dim() == 1:
            self.length_variates = self._mean_non_truncated.size(0)
            self.length_batch = 1
            self._mean_non_truncated = self._mean_non_truncated.unsqueeze(0)
            self._stddev_non_truncated = self._stddev_non_truncated.unsqueeze(0)
            self._low = self._low.unsqueeze(0)
            self._high = self._high.unsqueeze(0)
        elif self._mean_non_truncated.dim() == 2:
            self.length_variates = self._mean_non_truncated.size(1)
            self.length_batch = self._mean_non_truncated.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._standard_normal_dist = Normal(torch.zeros_like(self._mean_non_truncated), torch.ones_like(self._stddev_non_truncated))
        self._alpha = (self._low - self._mean_non_truncated) / self._stddev_non_truncated
        self._beta = (self._high - self._mean_non_truncated) / self._stddev_non_truncated
        self._standard_normal_cdf_alpha = self._standard_normal_dist.cdf(self._alpha)
        self._standard_normal_cdf_beta = self._standard_normal_dist.cdf(self._beta)
        self._Z = self._standard_normal_dist.cdf(self._beta) - self._standard_normal_dist.cdf(self._alpha)
        self._log_stddev_Z = torch.log(self._stddev_non_truncated * self._Z)
        self._mean = None
        self._variance = None
        super().__init__('TruncatedNormal', '_TruncatedNormal')

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        return torch.log(lb.mul(ub)) + self._standard_normal_dist.log_prob((value - self._mean_non_truncated) / self._stddev_non_truncated) - self._log_stddev_Z

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def mean_non_truncated(self):
        return self._mean_non_truncated

    @property
    def stddev_non_truncated(self):
        return self._stddev_non_truncated

    @property
    def variance_non_truncated(self):
        return self._stddev_non_truncated.pow(2)

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._mean_non_truncated + self._stddev_non_truncated * (self._standard_normal_dist.prob(self._alpha) - self._standard_normal_dist.prob(self._beta)) / self._Z
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            standard_normal_prob_alpha = self._standard_normal_dist.prob(self._alpha)
            standard_normal_prob_beta = self._standard_normal_dist.prob(self._beta)
            self._variance = self._stddev_non_truncated.pow(2) * (1 + ((self._alpha * standard_normal_prob_alpha - self._beta * standard_normal_prob_beta)/self._Z) - ((standard_normal_prob_alpha - standard_normal_prob_beta)/self._Z).pow(2))
        return self._variance

    def sample(self):
        shape = self._low.size()
        rand = util.to_variable(torch.zeros(shape).uniform_())

        attempt_count = 0
        ret = torch.zeros_like(rand).fill_(float('NaN'))
        while util.has_nan_or_inf(ret):
            attempt_count += 1
            if (attempt_count == 10000):
                print('Warning: trying to sample from the tail of a truncated normal distribution, which can take a long time. A more efficient implementation is pending.')
            ret = self._standard_normal_dist.icdf(self._standard_normal_cdf_alpha + rand * (self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha)) * self._stddev_non_truncated + self._mean_non_truncated
        return ret


# Temporary: this needs to be replaced by torch.distributions.Uniform when the new PyTorch version is released
class Uniform(Distribution):
    def __init__(self, low, high):
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        if self._low.dim() == 1:
            self.length_variates = self._low.size(0)
            self.length_batch = 1
            self._low = self._low.unsqueeze(0)
            self._high = self._high.unsqueeze(0)
        elif self._low.dim() == 2:
            self.length_variates = self._low.size(1)
            self.length_batch = self._low.size(0)
        self._mean = (self._high + self._low) / 2
        self._variance = (self._high - self._low).pow(2) / 12
        super().__init__('Uniform', '_Uniform')

    def __repr__(self):
        return 'Uniform(low: {}, high:{}, length_variates:{}, length_batch:{})'.format(self._low, self._high, self.length_variates, self.length_batch)

    def sample(self):
        shape = self._low.size()
        rand = util.to_variable(torch.zeros(shape).uniform_())
        return self._low + rand * (self._high - self._low)

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.lt(self._high).type_as(self._low)
        return torch.log(lb.mul(ub)) - torch.log(self._high - self._low)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high
