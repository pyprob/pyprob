import torch
from termcolor import colored

from . import Distribution, Exponential
from .. import util


class TruncatedExponential(Distribution):
    def __init__(self, rate_non_truncated, low, high):
        self._rate_non_truncated = util.to_tensor(rate_non_truncated)
        self._low = util.to_tensor(low)
        self._high = util.to_tensor(high)
        self._exponential_non_truncated = Exponential(self._rate_non_truncated)
        torch_expo = torch.distributions.Exponential(self._rate_non_truncated)
        self._renorm_factor = torch.ones(1) / (torch_expo.cdf(self._high) - torch_expo.cdf(self._low))
        self._rate = None
        self._batch_length = 1
        batch_shape = self._rate_non_truncated.size()
        event_shape = torch.Size()
        super().__init__(name='TruncatedExponential', address_suffix='TruncatedExponential', batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'TruncatedExponential(rate_non_truncated:{}, low:{}, high:{})'.format(self._rate_non_truncated, self._low, self._high)

    def log_prob(self, value, sum=False):
        value = util.to_tensor(value)
        lb = value.ge(self._low).type_as(self._low)
        ub = value.le(self._high).type_as(self._low)
        lp = torch.log(lb.mul(ub)) + torch.log(self._renorm_factor) + self._exponential_non_truncated.log_prob(value)
        if self._batch_length == 1:
            lp = lp.squeeze(0)
        if util.has_nan_or_inf(lp):
            print(colored('Warning: NaN, -Inf, or Inf encountered in TruncatedExponential log_prob.', 'red', attrs=['bold']))
            print('distribution', self)
            print('value', value)
            print('log_prob', lp)
            # lp = util.replace_inf(lp, colored('Warning: TruncatedNormal log_prob has inf, replacing with 0.', 'red', attrs=['bold']))
        return torch.sum(lp) if sum else lp


    def _icdf(self, value):
        value = util.to_tensor(value)
        return (torch.log(self._renorm_factor) - torch.log(self._renorm_factor * torch.exp(self._rate_non_truncated.mul(-1).mul(self._low)) - value)).div(self._rate_non_truncated)


    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def rate_non_truncated(self):
        return self._rate_non_truncated

    @property
    def mean_non_truncated(self):
        return self._exponential_non_truncated.mean

    @property
    def stddev_non_truncated(self):
        return self._exponential_non_truncated.stddev

    @property
    def variance_non_truncated(self):
        return self._exponential_non_truncated.variance

    @property
    def mean(self):
        if self._mean is None:
            # TODO
            #if self._batch_length == 1:
            #    self._mean = self._mean.squeeze(0)
            pass
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            # TODO
            #if self._batch_length == 1:
            #    self._variance = self._variance.squeeze(0)
            pass
        return self._variance

    def sample(self):
        # Adopted from the TruncatedNormal distribution
        shape = self._low.size()
        attempt_count = 0
        ret = util.to_tensor(torch.zeros(shape).fill_(float('NaN')))
        outside_domain = True
        while util.has_nan_or_inf(ret) or outside_domain:
            attempt_count += 1
            if (attempt_count == 10000):
                # Examples of better samplers: https://github.com/tensorflow/tensorflow/blob/502aad6f1934230911ed0d01515463c829bf845d/tensorflow/core/kernels/parameterized_truncated_normal_op.cc
                print('Warning: trying to sample from the tail of a truncated exponential distribution, which can take a long time. A more efficient implementation is pending.')
            rand = util.to_tensor(torch.zeros(shape).uniform_())
            ret = self._icdf(rand)
            lb = ret.ge(self._low).type_as(self._low)
            ub = ret.lt(self._high).type_as(self._low)
            outside_domain = (int(torch.sum(lb.mul(ub))) == 0)

        if self._batch_length == 1:
            ret = ret.squeeze(0)
        return ret
