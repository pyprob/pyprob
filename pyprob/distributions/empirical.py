import torch
import numpy as np
import copy
import shelve
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
import os
from termcolor import colored

from . import Distribution, Categorical
from .. import util


class Empirical(Distribution):
    def __init__(self, values=None, log_weights=None, weights=None, file_name=None, file_read_only=False, file_sync_timeout=1000, file_writeback=False, name='Empirical'):
        super().__init__(name)
        self._finalized = False
        self._read_only = file_read_only
        self._closed = False
        self._categorical = None
        self._log_weights = []
        self._length = 0
        self._uniform_weights = False
        if file_name is None:
            self._on_disk = False
            self._values = []
        else:
            self._on_disk = True
            if self._read_only:
                if not os.path.exists(file_name):
                    raise ValueError('File not found: {}'.format(file_name))
                flag = 'r'
            else:
                flag = 'c'
            self._file_name = file_name
            self._shelf = shelve.open(self._file_name, flag=flag, writeback=file_writeback)
            if 'name' in self._shelf:
                self.name = self._shelf['name']
            if 'log_weights' in self._shelf:
                self._log_weights = self._shelf['log_weights']
                self._file_last_key = self._shelf['last_key']
                self._length = len(self._log_weights)
            else:
                self._file_last_key = -1
            self._file_sync_timeout = file_sync_timeout
            self._file_sync_countdown = self._file_sync_timeout
            self.finalize()
        self._mean = None
        self._variance = None
        self._mode = None
        self._min = None
        self._max = None
        self._effective_sample_size = None
        if values is not None:
            if len(values) > 0:
                self.add_sequence(values, log_weights, weights)
                self.finalize()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if not self._closed:
            self.close()

    def __del__(self):
        if not self._closed:
            self.close()

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._length

    def close(self):
        if self._on_disk:
            self.finalize()
            if not self._closed:
                self._shelf.close()
                self._closed = True

    def copy(self, file_name=None):
        self._check_finalized()
        if self._on_disk:
            if file_name is None:
                print('Copying Empirical(file_name: {}) to Empirical(on memory)...'.format(self._file_name))
                return Empirical(values=self.get_values(), log_weights=self._log_weights, name=self.name)
            else:
                print('Copying Empirical(file_name: {}) to Empirical(file_name: {})...'.format(self._file_name, file_name))
                ret = Empirical(file_name=file_name, name=self.name)
                for i in range(self._length):
                    ret.add(value=self._shelf[str(i)], log_weight=self._log_weights[i])
                ret.finalize()
                return ret
        else:
            if file_name is None:
                print('Copying Empirical(on memory) to Empirical(on memory)...')
                return copy.copy(self)
            else:
                print('Copying Empirical(on memory) to Empirical(file_name: {})...'.format(file_name))
                return Empirical(values=self._values, log_weights=self._log_weights, file_name=file_name, name=self.name)

    def finalize(self):
        self._length = len(self._log_weights)
        self._categorical = torch.distributions.Categorical(logits=util.to_tensor(self._log_weights, dtype=torch.float64))
        if self._length > 0:
            self._uniform_weights = torch.eq(self._categorical.logits, self._categorical.logits[0]).all()
        else:
            self._uniform_weights = False
        if self._on_disk and not self._read_only:
            self._shelf['name'] = self.name
            self._shelf['log_weights'] = self._log_weights
            self._shelf['last_key'] = self._file_last_key
            self._shelf.sync()
        self._finalized = True

    def _check_finalized(self):
        if not self._finalized:
            raise RuntimeError('Empirical not finalized. Call finalize first.')

    def add(self, value, log_weight=None, weight=None):
        self._finalized = False
        self._mean = None
        self._variance = None
        self._mode = None
        self._min = None
        self._max = None
        self._effective_sample_size = None
        if log_weight is not None:
            self._log_weights.append(util.to_tensor(log_weight))
        elif weight is not None:
            self._log_weights.append(torch.log(util.to_tensor(weight)))
        else:
            self._log_weights.append(util.to_tensor(0.))

        if self._on_disk:
            self._file_last_key += 1
            self._shelf[str(self._file_last_key)] = value
            self._file_sync_countdown -= 1
            if self._file_sync_countdown == 0:
                self.finalize()
                self._file_sync_countdown = self._file_sync_timeout
        else:
            self._values.append(value)

    def add_sequence(self, values, log_weights=None, weights=None):
        if log_weights is not None:
            for i in range(len(values)):
                self.add(values[i], log_weight=log_weights[i])
        elif weights is not None:
            for i in range(len(values)):
                self.add(values[i], weight=weights[i])
        else:
            for i in range(len(values)):
                self.add(values[i])

    def rename(self, name):
        self.name = name
        if self._on_disk:
            self._shelf['name'] = self.name
        return self

    def _get_value(self, index):
        if self._on_disk:
            if index < 0:
                return self._get_value(self._length + index)
            return self._shelf[str(index)]
        else:
            return self._values[index]

    def _get_log_weight(self, index):
        return self._categorical.logits[index]

    def _get_weight(self, index):
        return self._categorical.probs[index]

    def get_values(self):
        self._check_finalized()
        if self._on_disk:
            return [self._shelf[str(i)] for i in range(self._length)]
        else:
            return self._values

    def sample(self, min_index=None, max_index=None):
        self._check_finalized()
        if self._uniform_weights:
            if min_index is None:
                min_index = 0
            if max_index is None:
                max_index = self._length - 1
            index = random.randint(min_index, max_index)
        else:
            if min_index is not None or max_index is not None:
                raise NotImplementedError()
            index = int(self._categorical.sample())
        return self._get_value(index)

    def __iter__(self):
        self._check_finalized()
        for i in range(self._length):
            yield self._get_value(i)

    def __getitem__(self, index):
        self._check_finalized()
        if isinstance(index, slice):
            if self._on_disk:
                raise NotImplementedError()
            return Empirical(values=self._values[index], log_weights=self._log_weights[index], name=self.name)
        else:
            return self._get_value(index)

    def expectation(self, func):
        self._check_finalized()
        ret = 0.
        if self._on_disk:
            for i in range(self._length):
                ret += util.to_tensor(func(self._shelf[str(i)]), dtype=torch.float64) * self._categorical.probs[i]
        else:
            if self._uniform_weights:
                ret = sum(map(func, self._values)) / self._length
            else:
                for i in range(self._length):
                    ret += util.to_tensor(func(self._values[i]), dtype=torch.float64) * self._categorical.probs[i]
        return util.to_tensor(ret)

    def map(self, func, *args, **kwargs):
        self._check_finalized()
        values = []
        for i in range(self._length):
            values.append(func(self._get_value(i)))
        return Empirical(values=values, log_weights=self._log_weights, name=self.name, *args, **kwargs)

    def filter(self, func, *args, **kwargs):
        self._check_finalized()
        if self.length == 0:
            return self
        filtered_values = []
        filtered_log_weights = []
        for i in range(self._length):
            value = self._get_value(i)
            if func(value):
                filtered_values.append(value)
                filtered_log_weights.append(self._get_log_weight(i))
        return Empirical(filtered_values, log_weights=filtered_log_weights, name=self.name, *args, **kwargs)

    def resample(self, num_samples, map_func=None, min_index=None, max_index=None, *args, **kwargs):
        self._check_finalized()
        # TODO: improve this with a better resampling algorithm
        if map_func is None:
            map_func = lambda x: x
        values = []
        message = 'Resampling{}{}...'.format('' if min_index is None else ', min_index: ' + str(min_index), '' if max_index is None else ', max_index: ' + str(max_index))
        util.progress_bar_init(message, num_samples, 'Samples')
        for i in range(num_samples):
            util.progress_bar_update(i)
            values.append(map_func(self.sample(min_index=None, max_index=None)))
        util.progress_bar_end()
        return Empirical(values=values, name=self.name, *args, **kwargs)

    def thin(self, num_samples, map_func=None, min_index=None, max_index=None, *args, **kwargs):
        self._check_finalized()
        if map_func is None:
            map_func = lambda x: x
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self.length
        step = max(1, math.floor((max_index - min_index) / num_samples))
        indices = range(min_index, max_index, step)
        values = []
        log_weights = []
        message = 'Thinning, step: {}{}{}...'.format(step, '' if min_index is None else ', min_index: ' + str(min_index), '' if max_index is None else ', max_index: ' + str(max_index))
        util.progress_bar_init(message, len(indices), 'Samples')
        for i in range(len(indices)):
            util.progress_bar_update(i)
            values.append(map_func(self._get_value(indices[i])))
            log_weights.append(self._get_log_weight(indices[i]))
        util.progress_bar_end()
        return Empirical(values=values, log_weights=log_weights, name=self.name, *args, **kwargs)

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
    def mode(self):
        self._check_finalized()
        if self._mode is None:
            if self._uniform_weights:
                counts = {}
                util.progress_bar_init('Computing mode...', self._length, 'Values')
                print(colored('Warning: weights are uniform and mode is correct only if values in Empirical are hashable', 'red', attrs=['bold']))
                for i in range(self._length):
                    util.progress_bar_update(i)
                    value = self._get_value(i)
                    if value in counts:
                        counts[value] += 1
                    else:
                        counts[value] = 1
                util.progress_bar_end()
                self._mode = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
            else:
                _, max_index = util.to_tensor(self._log_weights).max(-1)
                self._mode = self._get_value(int(max_index))
        return self._mode

    def arg_max(self, map_func):
        self._check_finalized()
        max_val = map_func(self._get_value(0))
        max_i = 0
        util.progress_bar_init('Computing arg_max...', self._length, 'Values')
        for i in range(self._length):
            util.progress_bar_update(i)
            val = map_func(self._get_value(i))
            if val >= max_val:
                max_val = val
                max_i = i
        util.progress_bar_end()
        return self._get_value(max_i)

    def arg_min(self, map_func):
        self._check_finalized()
        min_val = map_func(self._get_value(0))
        min_i = 0
        util.progress_bar_init('Computing arg_min...', self._length, 'Values')
        for i in range(self._length):
            util.progress_bar_update(i)
            val = map_func(self._get_value(i))
            if val <= min_val:
                min_val = val
                min_i = i
        util.progress_bar_end()
        return self._get_value(min_i)

    @property
    def effective_sample_size(self):
        self._check_finalized()
        if self._effective_sample_size is None:
            weights = self._categorical.probs
            self._effective_sample_size = 1. / weights.pow(2).sum()
            # log_weights = self._categorical.logits
            # self._effective_sample_size = torch.exp(2. * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2. * log_weights, dim=0))
        return self._effective_sample_size

    def unweighted(self, *args, **kwargs):
        self._check_finalized()
        return Empirical(values=self.get_values(), name=self.name, *args, **kwargs)

    def _find_min_max(self):
        try:
            sorted_values = sorted(map(float, self.get_values()))
            self._min = sorted_values[0]
            self._max = sorted_values[-1]
        except:
            raise RuntimeError('Cannot compute the minimum and maximum of values in this Empirical. Make sure the distribution is over values that are scalar or castable to scalar, e.g., a PyTorch tensor of one element.')

    @property
    def min(self):
        if self._min is None:
            self._find_min_max()
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._find_min_max()
        return self._max

    def combine_duplicates(self, *args, **kwargs):
        self._check_finalized()
        if self._on_disk:
            raise NotImplementedError()
        else:
            distribution = collections.defaultdict(float)
            # This can be simplified once PyTorch supports content-based hashing of tensors. See: https://github.com/pytorch/pytorch/issues/2569
            hashable = util.is_hashable(self._values[0])
            if hashable:
                for i in range(self.length):
                    found = False
                    for key, value in distribution.items():
                        if torch.equal(util.to_tensor(key), util.to_tensor(self._values[i])):
                            # Differentiability warning: values[i] is discarded here. If we need to differentiate through all values, the gradients of values[i] and key should be tied here.
                            distribution[key] = torch.logsumexp(torch.stack((value, self._log_weights[i])), dim=0)
                            found = True
                    if not found:
                        distribution[self._values[i]] = self._log_weights[i]
                values = list(distribution.keys())
                log_weights = list(distribution.values())
                return Empirical(values=values, log_weights=log_weights, name=self.name, *args, **kwargs)
            else:
                raise RuntimeError('The values in this Empirical as not hashable. Combining of duplicates not currently supported.')

    @staticmethod
    def combine(empirical_distributions, file_name=None):
        on_disk = empirical_distributions[0]._on_disk
        for dist in empirical_distributions:
            if dist._on_disk != on_disk:
                raise RuntimeError('Expecting all Empirical distributions to be on disk or in memory.')
            if not isinstance(dist, Empirical):
                raise TypeError('Combination is only supported between Empirical distributions.')

        if on_disk:
            if file_name is None:
                raise RuntimeError('Expecting a target file_name for the combined Empirical.')
            ret = Empirical(file_name=file_name)
            for dist in empirical_distributions:
                for i in range(dist._length):
                    ret.add(value=dist._shelf[str(i)], log_weight=dist._log_weights[i])
            ret.finalize()
            return ret
        else:
            values = []
            log_weights = []
            length = empirical_distributions[0].length
            for dist in empirical_distributions:
                if dist.length != length:
                    raise RuntimeError('Combination is only supported between Empirical distributions of equal length.')
                values += dist._values
                log_weights += dist._log_weights
            return Empirical(values=values, log_weights=log_weights, file_name=file_name)

    def values_numpy(self):
        self._check_finalized()
        try:  # This can fail in the case values are an iterable collection of non-numeric types (strings, etc.)
            return torch.stack(self.get_values()).cpu().numpy()
        except:
            try:
                return np.array(self.get_values())
            except:
                raise RuntimeError('Cannot convert values to numpy.')

    def weights_numpy(self):
        self._check_finalized()
        return util.to_numpy(self._categorical.probs)

    def log_weights_numpy(self):
        self._check_finalized()
        return util.to_numpy(self._categorical.logits)

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
