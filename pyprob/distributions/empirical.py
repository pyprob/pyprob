import torch
import numpy as np
import math
import random
import copy
from termcolor import colored
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import Distribution
from .. import util


class Empirical(Distribution):
    def __init__(self, values, log_weights=None, weights=None, sorted_by_weights=True, name='Empirical'):
        self.length = len(values)
        if isinstance(values, list):
            self.values = values
        else:
            self.values = [values[i] for i in range(self.length)]
        self.sorted_by_weights = sorted_by_weights
        if self.length == 0:
            super().__init__(name + ' (Empty)')
        else:
            if log_weights is not None:
                self.log_weights = util.to_tensor(log_weights, dtype=torch.float64).view(-1)
                self.weights = torch.exp(self.log_weights - torch.logsumexp(self.log_weights, dim=0))
                self._uniform_weights = torch.eq(self.weights, self.weights[0]).all()
            elif weights is not None:
                self.weights = util.to_tensor(weights, dtype=torch.float64).view(-1)
                self.weights = self.weights / self.weights.sum()
                self._uniform_weights = torch.eq(self.weights, self.weights[0]).all()
                self.log_weights = torch.log(self.weights)
            else:
                # assume uniform distribution if no log_weights or weights are given
                self.weights = util.to_tensor(torch.zeros(self.length), dtype=torch.float64).fill_(1./self.length)
                self.log_weights = torch.zeros(self.length).fill_(-math.log(self.length))
                self._uniform_weights = True

            if sorted_by_weights and not self._uniform_weights:
                self.weights, indices = torch.sort(self.weights, descending=True)
                log_weights = util.to_tensor([self.log_weights[int(i)] for i in indices], dtype=torch.float64).view(-1)
                self.values = [values[int(i)] for i in indices]

            self._weights_numpy = self.weights.cpu().numpy()
            self._weights_numpy_cumsum = (self._weights_numpy / self._weights_numpy.sum()).cumsum()
            self._mean = None
            self._variance = None
            self._mode = None
            self._min = None
            self._max = None
            self._effective_sample_size = None
            super().__init__(name)

    def __len__(self):
        return self.length

    def __repr__(self):
        return 'Empirical(name:{}, length:{})'.format(self.name, self.length)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Empirical(self.values[index], self.log_weights[index])
        else:
            return self.values[index]

    def __iter__(self):
        for i in range(len(self.values)):
            yield self.values[i], self.weights[i]

    def rename(self, name):
        self.name = name
        return self

    def sample(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            return random.choice(self.values)
        else:
            return util.fast_np_random_choice(self.values, self._weights_numpy_cumsum)

    def expectation(self, func):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            return util.to_tensor(sum(map(func, self.values)) / self.length)
        else:
            ret = 0.
            for i in range(self.length):
                ret += util.to_tensor(func(self.values[i]), dtype=torch.float64) * self.weights[i]
            return util.to_tensor(ret)

    def map(self, func):
        if self.length == 0:
            return self
        ret = copy.copy(self)
        ret.values = list(map(func, self.values))
        ret._mean = None
        ret._variance = None
        ret._mode = None
        ret._min = None
        ret._max = None
        ret._effective_sample_size = None
        return ret

    def filter(self, func):
        if self.length == 0:
            return self
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
            self._mean = self.expectation(lambda x: x)
        return self._mean

    @property
    def mode(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._uniform_weights:
            print(colored('Warning: weights are uniform and there is no unique mode.', 'red', attrs=['bold']))
        if self._mode is None:
            if self.sorted_by_weights:
                self._mode = self.values[0]
            else:
                _, max_index = self.weights.max(-1)
                self._mode = self.values[int(max_index)]
        return self._mode

    @property
    def variance(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._variance is None:
            mean = self.mean
            self._variance = self.expectation(lambda x: (x - mean)**2)
        return self._variance

    @property
    def effective_sample_size(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        if self._effective_sample_size is None:
            self._effective_sample_size = 1. / self.weights.pow(2).sum()
        return self._effective_sample_size

    def unweighted(self):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        return Empirical(self.values, name=self.name)

    def resample(self, samples):
        if self.length == 0:
            raise RuntimeError('Empirical distribution instance is empty.')
        # TODO: improve this with a better resampling algorithm
        return Empirical([self.sample() for i in range(samples)])

    @staticmethod
    def combine(empirical_distributions):
        for dist in empirical_distributions:
            if not isinstance(dist, Empirical):
                raise TypeError('Combination is only supported between Empirical distributions.')

        values = []
        log_weights = []
        length = empirical_distributions[0].length
        for dist in empirical_distributions:
            if dist.length != length:
                raise RuntimeError('Combination is only supported between Empirical distributions of equal length.')
            values += dist.values
            log_weights.append(dist.log_weights)
        return Empirical(values, torch.cat(log_weights))

    def values_numpy(self):
        try:  # This can fail in the case values are an iterable collection of non-numeric types (strings, etc.)
            return torch.stack(self.values).cpu().numpy()
        except:
            try:
                return np.array(self.values)
            except:
                raise RuntimeError('Cannot convert values to numpy.')

    def weights_numpy(self):
        return self._weights_numpy

    def plot_histogram(self, figsize=(10, 5), xlabel=None, ylabel='Frequency', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, density=1, *args, **kwargs):
        if not show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        values = self.values_numpy()
        weights = self.weights_numpy()
        plt.hist(values, weights=weights, density=density, *args, **kwargs)
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        if xlabel is None:
            xlabel = self.name
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fig.tight_layout()
        if file_name is not None:
            plt.savefig(file_name)
        if show:
            plt.show()
