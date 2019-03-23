import torch
from termcolor import colored

from . import Distribution, Normal
from .. import util


# Beware: clamp_mean_between_low_high=True prevents derivative computation with respect to mean when it's outside [low, high]
class TruncatedNormal(Distribution):
    def __init__(self, mean_non_truncated, stddev_non_truncated, low, high, clamp_mean_between_low_high=False):
        self._mean_non_truncated = util.to_tensor(mean_non_truncated)
        self._stddev_non_truncated = util.to_tensor(stddev_non_truncated)
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        if clamp_mean_between_low_high:
            self._mean_non_truncated = torch.max(torch.min(self._mean_non_truncated, self._high), self._low)
        if self._mean_non_truncated.dim() == 0:
            self._batch_length = 0
        elif self._mean_non_truncated.dim() == 1 or self._mean_non_truncated.dim() == 2:
            self._batch_length = self._mean_non_truncated.size(0)
        else:
            raise RuntimeError('Expecting 1d or 2d (batched) probabilities.')
        self._standard_normal_dist = Normal(util.to_tensor(torch.zeros_like(self._mean_non_truncated)), util.to_tensor(torch.ones_like(self._stddev_non_truncated)))
        self._alpha = (self._low - self._mean_non_truncated) / self._stddev_non_truncated
        self._beta = (self._high - self._mean_non_truncated) / self._stddev_non_truncated
        self._standard_normal_cdf_alpha = self._standard_normal_dist.cdf(self._alpha)
        self._standard_normal_cdf_beta = self._standard_normal_dist.cdf(self._beta)
        self._Z = self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha
        self._log_stddev_Z = torch.log(self._stddev_non_truncated * self._Z)
        self._mean = None
        self._variance = None
        batch_shape = self._mean_non_truncated.size()
        event_shape = torch.Size()
        super().__init__(name='TruncatedNormal', address_suffix='TruncatedNormal', batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'TruncatedNormal(mean_non_truncated:{}, stddev_non_truncated:{}, low:{}, high:{})'.format(self._mean_non_truncated, self._stddev_non_truncated, self._low, self._high)

    def log_prob(self, value, sum=False):
        value = util.to_tensor(value)
        # TODO: With the following handling of low and high bounds, the derivative is not correct for a value outside the truncation domain
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        lp = torch.log(lb.mul(ub)) + self._standard_normal_dist.log_prob((value - self._mean_non_truncated) / self._stddev_non_truncated) - self._log_stddev_Z
        if self._batch_length == 1:
            lp = lp.squeeze(0)
        if util.has_nan_or_inf(lp):
            print(colored('Warning: NaN, -Inf, or Inf encountered in TruncatedNormal log_prob.', 'red', attrs=['bold']))
            print('distribution', self)
            print('value', value)
            print('log_prob', lp)
            # lp = util.replace_inf(lp, colored('Warning: TruncatedNormal log_prob has inf, replacing with 0.', 'red', attrs=['bold']))
        return torch.sum(lp) if sum else lp

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
            if self._batch_length == 1:
                self._mean = self._mean.squeeze(0)
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            standard_normal_prob_alpha = self._standard_normal_dist.prob(self._alpha)
            standard_normal_prob_beta = self._standard_normal_dist.prob(self._beta)
            self._variance = self._stddev_non_truncated.pow(2) * (1 + ((self._alpha * standard_normal_prob_alpha - self._beta * standard_normal_prob_beta)/self._Z) - ((standard_normal_prob_alpha - standard_normal_prob_beta)/self._Z).pow(2))
            if self._batch_length == 1:
                self._variance = self._variance.squeeze(0)
        return self._variance

    def sample(self):
        shape = self._low.size()
        attempt_count = 0
        ret = util.to_tensor(torch.zeros(shape).fill_(float('NaN')))
        outside_domain = True
        while util.has_nan_or_inf(ret) or outside_domain:
            attempt_count += 1
            if (attempt_count == 10000):
                # Examples of better samplers: https://github.com/tensorflow/tensorflow/blob/502aad6f1934230911ed0d01515463c829bf845d/tensorflow/core/kernels/parameterized_truncated_normal_op.cc
                print('Warning: trying to sample from the tail of a truncated normal distribution, which can take a long time. A more efficient implementation is pending.')
            rand = util.to_tensor(torch.zeros(shape).uniform_())
            ret = self._standard_normal_dist.icdf(self._standard_normal_cdf_alpha + rand * (self._standard_normal_cdf_beta - self._standard_normal_cdf_alpha)) * self._stddev_non_truncated + self._mean_non_truncated
            lb = ret.ge(self._low).type_as(self._low)
            ub = ret.lt(self._high).type_as(self._low)
            outside_domain = (int(torch.sum(lb.mul(ub))) == 0)

        if self._batch_length == 1:
            ret = ret.squeeze(0)
        return ret
