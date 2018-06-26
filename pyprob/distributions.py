import torch
from torch.autograd import Variable
import torch.distributions
import copy
import collections
import math
import tarfile
import tempfile
import shutil
import os
import uuid
import numpy as np
import random
from threading import Thread
from termcolor import colored
from functools import reduce

from . import util, __version__


class Distribution(object):
    def __init__(self, name, address_suffix='', torch_dist=None):
        self.name = name
        self.address_suffix = address_suffix
        self._torch_dist = torch_dist

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            if self.length_batch > 1 and s.dim() == 1 and self.length_variates == 1:
                s = s.unsqueeze(1)
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value):
        if self._torch_dist is not None:
            value = util.to_variable(value)
            lp = self._torch_dist.log_prob(value)
            if lp.dim() == 2 and self.length_variates > 1:
                lp = util.safe_torch_sum(lp, dim=1)
            if self.length_batch > 1 and self.length_variates > 1:
                lp = lp.unsqueeze(1)
            return lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        value = util.to_variable(value)
        return torch.exp(self.log_prob(value))

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
            except AttributeError:  # This is because of the changing nature of PyTorch distributions. Should be removed when PyTorch stabilizes.
                return self._torch_dist.std.pow(2)
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        if self._torch_dist is not None:
            try:
                return self._torch_dist.stddev
            except AttributeError:  # This is because of the changing nature of PyTorch distributions. Should be removed when PyTorch stabilizes.
                return self._torch_dist.std
        else:
            return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    def save(self, file_name):
        data = {}
        data['distribution'] = self
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_distribution')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_distribution')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def load(file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_distribution')
            tar.extract('pyprob_distribution', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
            shutil.rmtree(tmp_dir)
        except:
            raise RuntimeError('Cannot load distribution.')

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded distribution: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded distribution: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        return data['distribution']


class Empirical(Distribution):
    def __init__(self, values, log_weights=None, weights=None, combine_duplicates=False, name='Empirical'):
        # print('Empirical initial_log_weights', log_weights)
        length = len(values)
        if length == 0:
            self.length = 0
            super().__init__(name + ' (Empty)')
        else:
            self._initial_log_weights = log_weights
            self._initial_weights = weights

            if log_weights is not None:
                log_weights = util.to_variable(log_weights).view(-1)
                weights = torch.exp(log_weights - util.log_sum_exp(log_weights))
            elif weights is not None:
                weights = util.to_variable(weights)
                weights = weights / weights.sum()
            else:
                # assume uniform distribution if no log_weights or weights are given
                # log_weights = util.to_variable(torch.zeros(length)).fill_(-math.log(length))
                weights = util.to_variable(torch.zeros(length).fill_(1./length))

            if isinstance(values, Variable) or torch.is_tensor(values):
                values = util.to_variable(values)
            elif isinstance(values, (list, tuple)):
                if isinstance(values[0], Variable) or torch.is_tensor(values[0]):
                    values = util.to_variable(values)

            self._initial_values = values

            if combine_duplicates:
                distribution = collections.defaultdict(float)
                # This can be simplified once PyTorch supports content-based hashing of tensors. See: https://github.com/pytorch/pytorch/issues/2569
                hashable = util.is_hashable(values[0])
                if hashable:
                    for i in range(length):
                        found = False
                        for key, value in distribution.items():
                            if torch.equal(util.to_variable(key), util.to_variable(values[i])):
                                # Differentiability warning: values[i] is discarded here. If we need to differentiate through all values, the gradients of values[i] and key should be tied here.
                                distribution[key] = value + weights[i]
                                found = True
                        if not found:
                            distribution[values[i]] = weights[i]
                    # else:
                    #     for i in range(length):
                    #         distribution[values[i]] += weights[i]
                    values = list(distribution.keys())
                    weights = list(distribution.values())
                    weights = torch.stack(weights)
            else:
                values = [values[i] for i in range(length)]

            self.length = len(values)

            self._uniform_weights = torch.eq(weights, weights[0]).all()
            if self._uniform_weights:
                self.weights = weights
                self.values = values
            else:
                self.weights, indices = torch.sort(weights, descending=True)
                self.values = [values[int(i)] for i in indices]
            self.weights_numpy = self.weights.data.cpu().numpy()
            self.weights_numpy_cumsum = (self.weights_numpy / self.weights_numpy.sum()).cumsum()
            # try:  # This can fail in the case values are an iterable collection of non-numeric types (strings, etc.)
            #     self.values_numpy = torch.stack(self.values).data.cpu().numpy()
            # except:
            #     try:
            #         self.values_numpy = np.array(self.values)
            #     except:
            #         self.values_numpy = None
            self._mean = None
            self._variance = None
            self._min = None
            self._max = None
            self._effective_sample_size = None
            # print('Empirical final_weights', self.weights)

            super().__init__(name)

    def __len__(self):
        return self.length

    def __repr__(self):
        try:
            return 'Empirical(name:{}, length:{}, mean:{}, stddev:{})'.format(self.name, self.length, self.mean, self.stddev)
        except RuntimeError:
            return 'Empirical(name:{}, length:{})'.format(self.name, self.length)

    def sample(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            return random.choice(self.values)
        else:
            return util.fast_np_random_choice(self.values, self.weights_numpy_cumsum)

    def expectation(self, func):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            # print('computing empirical expectation with uniform weights')
            return util.to_variable(sum(map(func, self.values)) / self.length)
        else:
            # print('computing empirical expectation with non-uniform weights')
            ret = 0.
            for i in range(self.length):
                # print('value {}: {}'.format(i, self.values[i]))
                # print('weight {}: {}'.format(i, self.weights[i]))
                ret += func(self.values[i]) * self.weights[i]
            return ret

    def map(self, func):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        ret = copy.copy(self)
        ret.values = list(map(func, self.values))
        ret._initial_values = list(map(func, self._initial_values))
        ret._mean = None
        ret._variance = None
        ret._min = None
        ret._max = None
        ret._mean = None
        ret._variance = None
        ret._effective_sample_size = None
        return ret

    def filter(self, func):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        filtered_values = []
        filtered_weights = []
        for i in range(len(self.values)):
            value = self.values[i]
            if func(value):
                filtered_values.append(value)
                filtered_weights.append(self.weights[i])
        return Empirical(filtered_values, weights=filtered_weights)

    @property
    def min(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._min is None:
            try:
                sorted_values = sorted(map(float, self.values))
                self._min = sorted_values[0]
                self._max = sorted_values[-1]
            except:
                raise RuntimeError('Cannot compute the minimum of values in this Empirical. Make sure the distribution is over values that are scalar or castable to scalar, e.g., a PyTorch tensor of one element.')
        return self._min

    @property
    def max(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._max is None:
            try:
                sorted_values = sorted(map(float, self.values))
                self._min = sorted_values[0]
                self._max = sorted_values[-1]
            except:
                raise RuntimeError('Cannot compute the maximum of values in this Empirical. Make sure the distribution is over values that are scalar or castable to scalar, e.g., a PyTorch tensor of one element.')
        return self._max

    @property
    def mean(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._mean is None:
            # print('computing empirical mean')
            self._mean = self.expectation(lambda x: x)
        return self._mean

    @property
    def effective_sample_size(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._effective_sample_size is None:
            self._effective_sample_size = 1. / self.weights.pow(2).sum()
        return self._effective_sample_size

    @property
    def mode(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            print(colored('Warning: weights are uniform and there is no unique mode.', 'red', attrs=['bold']))
        return self.values[0]  # Values are always sorted in decreasing weight

    @property
    def variance(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._variance is None:
            mean = self.mean
            self._variance = self.expectation(lambda x: (x - mean)**2)
        return self._variance

    def unweighted(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        return Empirical(self.values)

    def resample(self, samples):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        # TODO: improve this with a better resampling algorithm
        return Empirical([self.sample() for i in range(samples)])

    @staticmethod
    def combine(empirical_distributions, use_initial_values_and_weights=False):
        for dist in empirical_distributions:
            if not isinstance(dist, Empirical):
                raise TypeError('Combination is only supported between Empirical distributions.')
        if use_initial_values_and_weights:
            values = []
            weights = []
            log_weights = []
            for dist in empirical_distributions:
                values += dist._initial_values
                if dist._initial_weights is not None:
                    weights += dist._initial_weights
                if dist._initial_log_weights is not None:
                    log_weights += dist._initial_log_weights
            if len(weights) == len(values):
                return Empirical(values, weights=weights)
            elif len(log_weights) == len(values):
                return Empirical(values, log_weights=log_weights)
            elif len(log_weights) == 0 and len(weights) == 0:
                return Empirical(values)
            else:
                raise RuntimeError('Combination with use_initial_values_and_weights is only supported between Empirical distributions where all distributions initially have weights or all distributions initially have log_weights.')
        else:
            all_uniform = reduce(lambda x, y: x and y, [dist._uniform_weights for dist in empirical_distributions])
            if all_uniform:
                values = []
                for dist in empirical_distributions:
                    values += dist.values
                return Empirical(values)
            else:
                values = []
                weights = []
                for dist in empirical_distributions:
                    values += dist.values
                    weights.append(dist.weights * dist.length)
                return Empirical(values, weights=torch.cat(weights))


class Categorical(Distribution):
    def __init__(self, probs):
        self._probs = util.to_variable(probs)
        if self._probs.dim() == 1:
            self.length_variates = 1
            self.length_batch = 1
            self.length_categories = self._probs.size(0)
            self._probs = self._probs.unsqueeze(0)
        elif self._probs.dim() == 2:
            self.length_variates = 1
            self.length_batch = self._probs.size(0)
            self.length_categories = self._probs.size(1)
        else:
            raise ValueError('Expecting 1d or 2d (batched) probabilities.')
        self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        super().__init__('Categorical', 'Categorical(length_categories:{})'.format(self.length_categories), torch.distributions.Categorical(probs=self._probs))

    def __repr__(self):
        return 'Categorical(probs:{}, length_variates:{}, length_batch:{})'.format(self._probs, self.length_variates, self.length_batch)

    def __len__(self):
        return self.length_variates

    def sample(self):
        ret = self._torch_dist.sample().float()
        if self.length_batch == 1 and len(ret.size()) == 0:
            ret = ret.unsqueeze(0)
        return ret

    def log_prob(self, value):
        value = util.to_variable(value)
        value = util.to_variable(value).view(-1).long()
        ret = self._torch_dist.log_prob(value)
        if self.length_batch > 1:
            ret = ret.unsqueeze(1)
        return ret


class Mixture(Distribution):
    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length_distributions = len(distributions)
        if probs is None:
            self._probs = util.to_variable(torch.zeros(self.length_distributions).fill_(1/self.length_distributions))
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
        super().__init__('Mixture', 'Mixture({})'.format(', '.join([d.address_suffix for d in self._distributions])))

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value):
        value = util.to_variable(value).view(self.length_batch, 1)
        ret = util.log_sum_exp(torch.log(self._probs) + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).view(-1, self.length_batch).t())
        if self.length_batch == 1:
            ret = ret.squeeze(1)
        return ret

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
            if self.length_batch > 1:
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
            if self.length_batch > 1:
                self._variance = self._variance.unsqueeze(1)
        return self._variance


class Normal(Distribution):
    def __init__(self, mean, stddev):
        self._mean = util.to_variable(mean)
        self._stddev = util.to_variable(stddev)
        if self._mean.dim() == 0:
            self._mean = self._mean.unsqueeze(0)
            self._stddev = self._stddev.unsqueeze(0)
        if self._mean.dim() == 1:
            self.length_variates = self._mean.size(0)
            self.length_batch = 1
            if self._mean.size(0) > 1:
                self._mean = self._mean.unsqueeze(0)
                self._stddev = self._stddev.unsqueeze(0)
        elif self._mean.dim() == 2:
            self.length_variates = self._mean.size(1)
            self.length_batch = self._mean.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        super().__init__('Normal', 'Normal', torch.distributions.Normal(self._mean, self._stddev))

    def __repr__(self):
        return 'Normal(mean:{}, stddev:{}, length_variates:{}, length_batch:{})'.format(self._mean, self._stddev, self.length_variates, self.length_batch)

    # Won't be needed when the new PyTorch version is released
    def cdf(self, value):
        value = util.to_variable(value)
        return 0.5 * (1. + torch.erf((value - self._mean) * self._stddev.reciprocal() / math.sqrt(2)))

    # Won't be needed when the new PyTorch version is released
    def icdf(self, value):
        value = util.to_variable(value)
        return self._mean + self._stddev * torch.erfinv(2 * value - 1) * math.sqrt(2)


# Beware: clamp_mean_between_low_high=True prevents derivatives with respect to mean when it's outside [low, high]
class TruncatedNormal(Distribution):
    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high, clamp_mean_between_low_high=False):
        self._mean_non_truncated = util.to_variable(mean_non_truncated)
        self._stddev_non_truncated = util.to_variable(stddev_non_truncated)
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        if clamp_mean_between_low_high:
            self._mean_non_truncated = torch.max(torch.min(self._mean_non_truncated, self._high), self._low)
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
        self._Z = self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha
        self._log_stddev_Z = torch.log(self._stddev_non_truncated * self._Z)
        self._mean = None
        self._variance = None
        super().__init__('TruncatedNormal', 'TruncatedNormal')

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        #  TODO: With the following handling of low and high bounds, the derivative is not correct for a value outside the truncation domain
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        ret = torch.log(lb.mul(ub)) + self._standard_normal_dist.log_prob((value - self._mean_non_truncated) / self._stddev_non_truncated) - self._log_stddev_Z
        if self.length_batch == 1:
            ret = ret.squeeze(0)
        return ret

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
            if self.length_batch == 1:
                self._mean = self._mean.squeeze(0)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            standard_normal_prob_alpha = self._standard_normal_dist.prob(self._alpha)
            standard_normal_prob_beta = self._standard_normal_dist.prob(self._beta)
            self._variance = self._stddev_non_truncated.pow(2) * (1 + ((self._alpha * standard_normal_prob_alpha - self._beta * standard_normal_prob_beta)/self._Z) - ((standard_normal_prob_alpha - standard_normal_prob_beta)/self._Z).pow(2))
            if self.length_batch == 1:
                self._variance = self._variance.squeeze(0)
        return self._variance

    def sample(self):
        shape = self._low.size()
        attempt_count = 0
        ret = torch.zeros(shape).fill_(float('NaN'))
        outside_domain = True
        while util.has_nan_or_inf(ret) or outside_domain:
            attempt_count += 1
            if (attempt_count == 10000):
                print('Warning: trying to sample from the tail of a truncated normal distribution, which can take a long time. A more efficient implementation is pending.')
            rand = util.to_variable(torch.zeros(shape).uniform_())
            ret = self._standard_normal_dist.icdf(self._standard_normal_cdf_alpha + rand * (self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha)) * self._stddev_non_truncated + self._mean_non_truncated
            lb = ret.ge(self._low).type_as(self._low)
            ub = ret.lt(self._high).type_as(self._low)
            outside_domain = (float(util.safe_torch_sum(lb.mul(ub))) == 0.)

        if self.length_batch == 1:
            ret = ret.squeeze(0)
        return ret


# Temporary: this needs to be based on torch.distributions.Uniform when the new PyTorch version is released
class Uniform(Distribution):
    def __init__(self, low=0, high=1):
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        if self._low.dim() == 1:
            self.length_variates = self._low.size(0)
            self.length_batch = 1
        elif self._low.dim() == 2:
            self.length_variates = self._low.size(1)
            self.length_batch = self._low.size(0)
        self._range = self._high - self._low
        self._log_range = torch.log(self._range)
        self._mean = (self._high + self._low) / 2
        self._variance = (self._high - self._low).pow(2) / 12
        super().__init__('Uniform', 'Uniform')

    def __repr__(self):
        return 'Uniform(low: {}, high:{}, length_variates:{}, length_batch:{})'.format(self._low, self._high, self.length_variates, self.length_batch)

    def sample(self):
        shape = self._low.size()
        accept = False
        while not accept:
            rand = torch.zeros(shape).uniform_()
            if (rand.gt(0.).mul(rand.lt(1.))).sum() == len(rand):
                accept = True
        ret = self._low + util.to_variable(rand) * self._range
        if self.length_batch == 1:
            ret = ret.squeeze(0)
        if len(ret.size()) == 0:
            ret = ret.unsqueeze(0)
        return ret

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        lb = value.gt(self._low).type_as(self._low)
        ub = value.lt(self._high).type_as(self._low)
        ret = torch.log(lb.mul(ub)) - self._log_range
        if self.length_batch == 1:
            ret = ret.squeeze(0)
        return ret

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


# Temporary: this needs to be based on torch.distributions.Poisson when the new PyTorch version is released
class Poisson(Distribution):
    def __init__(self, rate):
        self._rate = util.to_variable(rate)
        if self._rate.dim() == 1:
            self.length_variates = self._rate.size(0)
            self.length_batch = 1
            if self._rate.size(0) > 1:
                self._rate = self._rate.unsqueeze(0)
        elif self._rate.dim() == 2:
            self.length_variates = self._rate.size(1)
            self.length_batch = self._rate.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._rate_numpy = self._rate.data.cpu().numpy()
        super().__init__('Poisson', 'Poisson')

    def __repr__(self):
        return 'Poisson(rate: {}, length_variates:{}, length_batch:{})'.format(self._rate, self.length_variates, self.length_batch)

    def sample(self):
        ret = util.to_variable(np.random.poisson(self._rate_numpy))
        # if self.length_batch == 1:
        #     ret = ret.squeeze(0)
        # elif self.length_batch > 1 and ret.dim() == 1 and self.length_variates == 1:
        #     ret = ret.unsqueeze(1)
        return ret

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        lp = (self._rate.log() * value) - self._rate - (value + 1).lgamma()
        if lp.dim() == 2 and self.length_variates > 1:
            lp = util.safe_torch_sum(lp, dim=1)
        if self.length_batch > 1 and self.length_variates > 1:
            lp = lp.unsqueeze(1)
        return lp

    @property
    def mean(self):
        return self._rate

    @property
    def variance(self):
        return self._rate

    @property
    def rate(self):
        return self._rate


class Kumaraswamy(Distribution):
    def __init__(self, shape1, shape2, low=0, high=1):
        self._shape1 = util.to_variable(shape1)
        self._shape2 = util.to_variable(shape2)
        self._low = util.to_variable(low)
        self._high = util.to_variable(high)
        self._range = self._high - self._low
        self._log_range = torch.log(self._range)
        if self._shape1.dim() == 1:
            self.length_variates = self._shape1.size(0)
            self.length_batch = 1
            if self._shape1.size(0) > 1:
                self._shape1 = self._shape1.unsqueeze(0)
                self._shape2 = self._shape2.unsqueeze(0)
        elif self._shape1.dim() == 2:
            self.length_variates = self._shape1.size(1)
            self.length_batch = self._shape1.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._shape1_recip = 1. / self._shape1
        self._shape2_recip = 1. / self._shape2
        self._standard_uniform = Uniform(torch.zeros(self.length_batch, self.length_variates), torch.ones(self.length_batch, self.length_variates))
        self._mean = None
        self._variance = None
        self._epsilon = util.to_variable(1e-5)
        super().__init__('Kumaraswamy', 'Kumaraswamy')

    def __repr__(self):
        return 'Kumaraswamy(shape1:{}, shape2:{}, low:{}, high:{}, length_variates:{}, length_batch:{})'.format(self._shape1, self._shape2, self._low, self._high, self.length_variates, self.length_batch)

    def sample(self):
        accept = False
        while not accept:
            s = (1. - ((1. - self._standard_uniform.sample()) ** self._shape2_recip)) ** self._shape1_recip
            if int((s.gt(0.).mul(s.lt(1.))).sum().data.cpu()) == len(s):
                accept = True
        # s = torch.min(torch.max(self._epsilon, s), 1 - self._epsilon)
        return self._low + self._range * s

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self._low + self._range * torch.exp(torch.log(self._shape2) + torch.lgamma(1. + 1./self._shape1) + torch.lgamma(self._shape2) - torch.lgamma(1. + self._shape2 + 1./self._shape1))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            self._variance = self._range * self._range * torch.exp(torch.log(self._shape2) + torch.lgamma(1. + 2./self._shape1) + torch.lgamma(self._shape2) - torch.lgamma(1. + self._shape2 + 2./self._shape1))
            self._variance -= self.mean ** 2.
        return self._variance

    def log_prob(self, value):
        value = util.to_variable(value)
        value = value.view(self.length_batch, self.length_variates)
        value = (value - self._low) / self._range
        lp = -self._log_range + torch.log(self._shape1) + torch.log(self._shape2) + (self._shape1 - 1.) * torch.log(value) + (self._shape2 - 1.) * torch.log(1. - value**self._shape1)
        if lp.dim() == 2 and self.length_variates > 1:
            lp = util.safe_torch_sum(lp, dim=1)
        if self.length_batch == 1:
            lp = lp.squeeze(0)
        # if self.length_batch > 1 and self.length_variates > 1:
        #     lp = lp.unsqueeze(1)
        # if util.has_nan_or_inf(lp):
        lp[(lp != lp).detach()] = float('-inf')
        return lp
