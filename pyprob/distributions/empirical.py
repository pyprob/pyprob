import torch
import numpy as np
import copy
# import shelve
import collections
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import math
import os
import enum
import yaml
import warnings
from sklearn import mixture

from . import Distribution, Normal, Mixture
from .. import util
from ..trace import Trace, Variable


class EmpiricalType(enum.Enum):
    MEMORY = 1
    FILE = 2
    CONCAT_MEMORY = 3
    CONCAT_FILE = 4


class Empirical(Distribution):
    def __init__(self, values=None, log_weights=None, weights=None, file_name=None, file_read_only=False, file_sync_timeout=25, file_writeback=False, concat_empiricals=None, concat_empirical_file_names=None, name='Empirical'):
        super().__init__(name)
        self._finalized = False
        self._closed = False
        self._read_only = file_read_only
        if self._read_only:
            if file_name is None:
                raise ValueError('Cannot set file_read_only=True when there is no file_name argument')
            if not os.path.exists(file_name):
                raise ValueError('File not found: {}'.format(file_name))
            shelf_flag = 'r'
        else:
            shelf_flag = 'c'
        self._file_name = file_name
        self._categorical = None
        self.log_weights = []
        self._length = 0
        self._uniform_weights = False
        self._type = None
        self._metadata = OrderedDict()
        if concat_empiricals is not None or concat_empirical_file_names is not None:
            if concat_empiricals is not None:
                if type(concat_empiricals) == list:
                    if type(concat_empiricals[0]) == Empirical:
                        self._concat_empiricals = concat_empiricals
                    else:
                        raise TypeError('Expecting concat_empiricals to be a list of Empiricals.')
                else:
                    raise TypeError('Expecting concat_empiricals to be a list of Empiricals.')
            else:
                if type(concat_empirical_file_names) == list:
                    if type(concat_empirical_file_names[0]) == str:
                        concat_empirical_file_names = list(map(os.path.abspath, concat_empirical_file_names))
                        self._concat_empiricals = [Empirical(file_name=f, file_read_only=True) for f in concat_empirical_file_names]
                    else:
                        raise TypeError('Expecting concat_empirical_file_names to be a list of file names.')
                else:
                    raise TypeError('Expecting concat_empirical_file_names to be a list of file names.')
            self._concat_cum_sizes = np.cumsum([emp.length for emp in self._concat_empiricals])
            self._length = self._concat_cum_sizes[-1]
            self.log_weights = torch.cat([util.to_tensor(emp.log_weights) for emp in self._concat_empiricals])
            self._categorical = torch.distributions.Categorical(logits=util.to_tensor(self.log_weights, dtype=torch.float64))
            self._check_uniform_weights()
            weights = self._categorical.probs
            self._effective_sample_size = 1. / weights.pow(2).sum()
            name = 'Concatenated empirical, length: {:,}, ESS: {:,.2f}'.format(self._length, self._effective_sample_size)
            # self._metadata.append('Begin concatenate empiricals ({})'.format(len(self._concat_empiricals)))
            # for i, emp in enumerate(self._concat_empiricals):
            #     self._metadata.append('Begin source empirical ({}/{})'.format(i+1, len(self._concat_empiricals)))
            #     self._metadata.extend(emp._metadata)
            #     self._metadata.append('End source empirical ({}/{})'.format(i+1, len(self._concat_empiricals)))
            # self._metadata.append('End concatenate empiricals ({})'.format(len(self._concat_empiricals)))
            self.add_metadata(op='concat', num_empiricals=len(self._concat_empiricals), metadata_empiricals=[emp._metadata for emp in self._concat_empiricals])
            self.rename(name)
            self._finalized = True
            self._read_only = True
            if file_name is None:
                self._type = EmpiricalType.CONCAT_MEMORY
            else:
                if concat_empirical_file_names is None:
                    raise ValueError('Expecting concat_empirical_file_names to write a concatenated empirical file.')
                if shelf_flag == 'r':
                    raise RuntimeError('Empirical file already exists, cannot write new concatenated Empirical: {}'.format(self._file_name))
                else:
                    self._type = EmpiricalType.CONCAT_FILE
                    self._shelf = util.open_shelf(self._file_name, flag=shelf_flag, writeback=False)
                    self._shelf['concat_empirical_file_names'] = concat_empirical_file_names
                    self._shelf['name'] = self.name
                    self._shelf['metadata'] = self._metadata
                    self._shelf.close()
        else:
            if file_name is None:
                self._type = EmpiricalType.MEMORY
                self.values = []
            else:
                self._shelf = util.open_shelf(self._file_name, flag=shelf_flag, writeback=file_writeback)
                if 'concat_empirical_file_names' in self._shelf:
                    self._type = EmpiricalType.CONCAT_FILE
                    concat_empirical_file_names = self._shelf['concat_empirical_file_names']
                    self._concat_empiricals = [Empirical(file_name=f, file_read_only=True) for f in concat_empirical_file_names]
                    self._concat_cum_sizes = np.cumsum([emp.length for emp in self._concat_empiricals])
                    self._length = self._concat_cum_sizes[-1]
                    self.log_weights = torch.cat([util.to_tensor(emp.log_weights) for emp in self._concat_empiricals])
                    self._categorical = torch.distributions.Categorical(logits=util.to_tensor(self.log_weights, dtype=torch.float64))
                    self._check_uniform_weights()
                    self.name = self._shelf['name']
                    if 'metadata' in self._shelf:
                        self._metadata = self._shelf['metadata']
                    self._file_last_key = -1
                    self._finalized = True
                    self._read_only = True
                else:
                    self._type = EmpiricalType.FILE
                    if 'name' in self._shelf:
                        self.name = self._shelf['name']
                    if 'metadata' in self._shelf:
                        self._metadata = self._shelf['metadata']
                    if 'log_weights' in self._shelf:
                        self.log_weights = self._shelf['log_weights']
                        self._file_last_key = self._shelf['last_key']
                        self._length = len(self.log_weights)
                    else:
                        self._file_last_key = -1
                    self._file_sync_timeout = file_sync_timeout
                    self._file_sync_countdown = self._file_sync_timeout
                    self.finalize()
        self._mean = None
        self._variance = None
        self._skewness = None
        self._kurtosis = None
        self._mode = None
        self._median = None
        self._min = None
        self._max = None
        self._effective_sample_size = None
        self.add_metadata(name=self.name)
        if values is not None:
            if len(values) > 0:
                self.add_sequence(values, log_weights, weights)
                self.finalize()

        # if self._type == EmpiricalType.FILE or self._type == EmpiricalType.CONCAT_FILE:
        #     if not util.check_gnu_dbm():
        #         warnings.warn('Empirical distributions on disk may perform slow because GNU DBM is not available. Please install and configure gdbm library for Python for better speed.')

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self, '_closed'):
            if not self._closed:
                self.close()

    def __del__(self):
        if hasattr(self, '_closed'):
            if not self._closed:
                self.close()

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._length

    @property
    def weights(self):
        return self._categorical.probs

    @property
    def metadata(self):
        return self._metadata

    def add_metadata(self, **kwargs):
        self._metadata['{}'.format(len(self._metadata))] = kwargs

    def close(self):
        if self._type == EmpiricalType.FILE:
            self.finalize()
            if not self._closed:
                if not self._read_only:
                    self._shelf.close()
                self._closed = True

    def copy(self, file_name=None):
        self._check_finalized()
        if self._type == EmpiricalType.FILE:
            if file_name is None:
                status = 'Copy Empirical(file_name: {}) to Empirical(memory)'.format(self._file_name)
                print(status)
                ret = Empirical(values=self.get_values(), log_weights=self.log_weights, name=self.name)
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='copy', source='Empirical(file_name: {})'.format(self._file_name), target='Empirical(memory)')
                return ret
            else:
                status = 'Copy Empirical(file_name: {}) to Empirical(file_name: {})'.format(self._file_name, file_name)
                print(status)
                ret = Empirical(file_name=file_name, name=self.name)
                for i in range(self._length):
                    ret.add(value=self._get_value(i), log_weight=self.log_weights[i])
                ret.finalize()
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='copy', source='Empirical(file_name: {})'.format(self._file_name), target='Empirical(file_name: {})'.format(file_name))
                return ret
        elif self._type == EmpiricalType.MEMORY:
            if file_name is None:
                status = 'Copy Empirical(memory) to Empirical(memory)'
                print(status)
                ret = copy.copy(self)
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='copy', source='Empirical(memory)', target='Empirical(memory)')
                return ret
            else:
                status = 'Copy Empirical(memory) to Empirical(file_name: {})'.format(file_name)
                print(status)
                ret = Empirical(values=self.values, log_weights=self.log_weights, file_name=file_name, name=self.name)
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='copy', source='Empirical(memory)', target='Empirical(file_name: {})'.format(file_name))
                return ret
        else:
            raise NotImplementedError('Not implemented for type: {}'.format(str(self._type)))

    def _check_uniform_weights(self):
        if self._length > 0:
            self._uniform_weights = torch.eq(self._categorical.logits, self._categorical.logits[0]).all()
        else:
            self._uniform_weights = False

    def from_distribution(distribution, num_samples):
        return Empirical([distribution.sample() for _ in range(num_samples)])

    def finalize(self):
        self._length = len(self.log_weights)
        self._categorical = torch.distributions.Categorical(logits=util.to_tensor(self.log_weights, dtype=torch.float64))
        self.add_metadata(op='finalize', length=self._length)
        self._check_uniform_weights()
        if self._type == EmpiricalType.FILE and not self._read_only:
            self._shelf['name'] = self.name
            self._shelf['metadata'] = self._metadata
            self._shelf['log_weights'] = self.log_weights
            self._shelf['last_key'] = self._file_last_key
            self._shelf.sync()
        self._finalized = True

    def _check_finalized(self):
        if not self._finalized:
            raise RuntimeError('Empirical not finalized. Call finalize first.')

    def add(self, value, log_weight=None, weight=None):
        if self._read_only:
            raise RuntimeError('Empirical is read-only.')
        self._finalized = False
        self._mean = None
        self._variance = None
        self._mode = None
        self._min = None
        self._max = None
        self._effective_sample_size = None
        if log_weight is not None:
            self.log_weights.append(util.to_tensor(log_weight))
        elif weight is not None:
            self.log_weights.append(torch.log(util.to_tensor(weight)))
        else:
            self.log_weights.append(util.to_tensor(0.))

        if self._type == EmpiricalType.FILE:
            self._file_last_key += 1
            self._shelf[str(self._file_last_key)] = value
            self._file_sync_countdown -= 1
            if self._file_sync_countdown == 0:
                self.finalize()
                self._file_sync_countdown = self._file_sync_timeout
        else:
            self.values.append(value)

    def add_sequence(self, values, log_weights=None, weights=None):
        if self._read_only:
            raise RuntimeError('Empirical is read-only.')
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
        self.add_metadata(op='rename', name=name)
        self.name = name
        if self._type == EmpiricalType.FILE:
            self._shelf['name'] = self.name
        return self

    def _get_value(self, index):
        if self._type == EmpiricalType.MEMORY:
            return self.values[index]
        elif self._type == EmpiricalType.FILE:
            if index < 0:
                index = self._length + index
            return self._shelf[str(index)]
        else:  # CONCAT_MEMORY or CONCAT_FILE
            emp_index = self._concat_cum_sizes.searchsorted(index, 'right')
            if emp_index > 0:
                index = index - self._concat_cum_sizes[emp_index - 1]
            return self._concat_empiricals[emp_index]._get_value(index)

    def _get_log_weight(self, index):
        return self._categorical.logits[index]

    def _get_weight(self, index):
        return self._categorical.probs[index]

    def get_values(self):
        self._check_finalized()
        if self._type == EmpiricalType.MEMORY:
            return self.values
        elif self._type == EmpiricalType.FILE:
            return [self._shelf[str(i)] for i in range(self._length)]
        else:
            raise NotImplementedError('Not implemented for type: {}'.format(str(self._type)))

    def sample(self, min_index=None, max_index=None):  # min_index is inclusive, max_index is exclusive
        self._check_finalized()
        if self._uniform_weights:
            if min_index is None:
                min_index = 0
            if max_index is None:
                max_index = self._length
            index = random.randint(min_index, max_index-1)  # max_index-1 is because random.randint treats upper bound as inclusive and we define max_index as exclusive
            return self._get_value(index)
        else:
            if (min_index is None or min_index == 0) and (max_index is None or max_index == self.length):
                # We don't need to slice
                index = int(self._categorical.sample())
                return self._get_value(index)
            else:
                # We need to slice
                return self[min_index:max_index].sample()

    def __iter__(self):
        self._check_finalized()
        for i in range(self._length):
            yield self._get_value(i)

    def __getitem__(self, index):
        self._check_finalized()
        if isinstance(index, slice):
            if self._type == EmpiricalType.MEMORY:
                ret = Empirical(values=self.values[index], log_weights=self.log_weights[index], name=self.name)
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='slice', index='{}'.format(index))
                return ret
            else:
                min_index = index.start
                max_index = index.stop
                if min_index is None:
                    min_index = 0
                elif min_index < -self._length:
                    min_index = 0
                elif min_index < 0:
                    min_index += self._length
                if max_index is None:
                    max_index = self._length
                elif max_index > self._length:
                    max_index = self._length
                elif max_index < -self._length:
                    max_index = 0
                elif max_index < 0:
                    max_index += self._length
                return self.map(lambda x: x, min_index=min_index, max_index=max_index)
        elif isinstance(index, int):
            return self._get_value(index)
        elif isinstance(index, str):
            if isinstance(self[0], Trace):
                return self.map(lambda trace: trace[index])
            else:
                raise RuntimeError('A string index can only be used with an Empirical that contains execution traces.')
        else:
            raise RuntimeError('Cannot use the given value ({}) as index'.format(index))

    def expectation(self, func):
        self._check_finalized()
        ret = 0.
        if self._type == EmpiricalType.MEMORY:
            if self._uniform_weights:
                ret = sum(map(func, self.values)) / self._length
            else:
                for i in range(self._length):
                    ret += util.to_tensor(func(self.values[i]), dtype=torch.float64) * self._categorical.probs[i]
        elif self._type == EmpiricalType.FILE:
            for i in range(self._length):
                ret += util.to_tensor(func(self._shelf[str(i)]), dtype=torch.float64) * self._categorical.probs[i]
        else:  # CONCAT_MEMORY or CONCAT_FILE
            for i in range(self._length):
                ret += util.to_tensor(func(self._get_value(i)), dtype=torch.float64) * self._categorical.probs[i]
        return util.to_tensor(ret)

    # Idea by Giacomo Acciarini, August 2020
    def reobserve(self, likelihood_funcs=None, observe=None, likelihood_importance=1., min_index=None, max_index=None, file_name=None):
        if len(self) == 0:
            return self
        self._check_finalized()
        if not isinstance(self[0], Trace):
            raise RuntimeError('Reobserve can only be used with Empiricals that contain execution traces.')
        last_op = list(self.metadata.values())[-1]
        can_reobserve = False
        if 'op' in last_op:
            if (last_op['op'] == 'posterior') and ('IMPORTANCE_SAMPLING' in last_op['inference_engine']):
                can_reobserve = True
        if not can_reobserve:
            warnings.warn('Reobserve should ideally be used immediately following a posterior with an importance-sampling-based inference engine. Metadata of the distribution indicates the last operation was not such a posterior: {}'.format(self.metadata))
        if observe is None:
            observe = {}
        else:
            warnings.warn('Updating observed values with the ones supplied. If any part of the model code is conditional on the observed value, the resulting posterior will not be correct.')
        if likelihood_funcs is None:
            likelihood_funcs = {}
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self._length
        indices = range(min_index, max_index)
        status = 'Reobserve, min_index: {}, max_index: {}'.format(min_index, max_index)
        util.progress_bar_init(status, len(indices), 'Values')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in range(min_index, max_index):
            util.progress_bar_update(i)
            trace = self._get_value(i)
            new_trace = Trace()
            for v in trace.variables:
                if v.observable:
                    # print('Observable variable with name: {}'.format(v.name))
                    if v.name in observe:
                        value = observe[v.name]
                        # print('Observe new value: {}'.format(value))
                        observed = True
                    elif v.observed:
                        value = v.value
                        # print('Observe existing value: {}'.format(value))
                        observed = True
                    else:
                        # print('Not observed')
                        value = v.value
                        observed = False
                    if v.name in likelihood_funcs:
                        likelihood_func = likelihood_funcs[v.name]
                        distribution = likelihood_func(v, trace)
                        if value is None:
                            log_prob = None
                            log_importance_weight = None
                        else:
                            log_prob = likelihood_importance * distribution.log_prob(value, sum=True)
                            log_importance_weight = float(log_prob)
                    else:
                        distribution = v.distribution
                        log_prob = v.log_prob
                        log_importance_weight = v.log_importance_weight

                    address_base = v.address_base
                    address = v.address
                    instance = v.instance
                    control = v.control
                    name = v.name
                    reused = v.reused
                    tagged = v.tagged
                    v = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, control=control, name=name, observed=observed, reused=reused, tagged=tagged)
                new_trace.add(v)
            new_trace.end(result=trace.result, execution_time_sec=trace.execution_time_sec)
            ret.add(new_trace, new_trace.log_importance_weight)
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='reobserve', length=len(self), observe=observe, likelihood_func=util.get_source(likelihood_func))
        return ret

    def reweight(self, func, min_index=None, max_index=None, file_name=None):
        if len(self) == 0:
            return self
        self._check_finalized()
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self._length
        indices = range(min_index, max_index)
        status = 'Reweight, min_index: {}, max_index: {}'.format(min_index, max_index)
        util.progress_bar_init(status, len(indices), 'Values')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in range(min_index, max_index):
            util.progress_bar_update(i)
            v = self._get_value(i)
            ret.add(v, func(v))
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='reweight', length=len(self), func=util.get_source(func))
        return ret

    def map(self, func, min_index=None, max_index=None, file_name=None):
        if len(self) == 0:
            return self
        self._check_finalized()
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self._length
        indices = range(min_index, max_index)
        status = 'Map, min_index: {}, max_index: {}'.format(min_index, max_index)
        util.progress_bar_init(status, len(indices), 'Values')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in range(min_index, max_index):
            util.progress_bar_update(i)
            ret.add(func(self._get_value(i)), self._get_log_weight(i))
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='map', length=len(self), func=util.get_source(func))
        return ret

    def filter(self, func, min_index=None, max_index=None, file_name=None):
        self._check_finalized()
        if self.length == 0:
            return self
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self._length
        indices = range(min_index, max_index)
        status = 'Filter, min_index: {}, max_index: {}'.format(min_index, max_index)
        util.progress_bar_init(status, len(indices), 'Values')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in indices:
            util.progress_bar_update(i)
            value = self._get_value(i)
            if func(value):
                ret.add(value, self._get_log_weight(i))
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='filter', length=len(self), length_after=len(ret), func=util.get_source(func))
        ret.finalize()
        return ret

    def resample(self, num_samples, map_func=None, min_index=None, max_index=None, file_name=None):  # min_index is inclusive, max_index is exclusive
        self._check_finalized()
        # TODO: improve this with a better resampling algorithm
        if map_func is None:
            map_func = lambda x: x
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self.length
        ess_before_resample = float(self.effective_sample_size)
        status = 'Resample, num_samples: {}, min_index: {}, max_index: {}, ess_before_resample: {}'.format(num_samples, min_index, max_index, ess_before_resample)
        util.progress_bar_init(status, num_samples, 'Samples')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in range(num_samples):
            util.progress_bar_update(i)
            ret.add(self.sample(min_index=min_index, max_index=max_index))
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='resample', length=len(self), num_samples=int(num_samples), min_index=int(min_index), max_index=int(max_index), ess_before=ess_before_resample)
        return ret

    def thin(self, num_samples, map_func=None, min_index=None, max_index=None, file_name=None):  # min_index is inclusive, max_index is exclusive
        self._check_finalized()
        if map_func is None:
            map_func = lambda x: x
        if min_index is None:
            min_index = 0
        if max_index is None:
            max_index = self.length
        step = max(1, math.floor((max_index - min_index) / num_samples))
        indices = range(min_index, max_index, step)
        status = 'Thin, num_samples: {}, step: {}, min_index: {}, max_index: {}'.format(num_samples, step, min_index, max_index)
        util.progress_bar_init(status, len(indices), 'Values')
        ret = Empirical(name=self.name, file_name=file_name)
        for i in range(len(indices)):
            util.progress_bar_update(i)
            v = map_func(self._get_value(indices[i]))
            lw = self._get_log_weight(indices[i])
            ret.add(v, lw)
        ret.finalize()
        util.progress_bar_end()
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='thin', length=len(self), num_samples=int(num_samples), step=int(step), min_index=int(min_index), max_index=int(max_index))
        return ret

    @property
    def weighted(self):
        return not self._uniform_weights

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
    def skewness(self):
        if self._skewness is None:
            self._skewness = self.expectation(lambda x: ((x-self.mean)/self.stddev)**3)
        return self._skewness

    @property
    def kurtosis(self):
        if self._kurtosis is None:
            self._kurtosis = self.expectation(lambda x: ((x-self.mean)/self.stddev)**4)
        return self._kurtosis

    @property
    def mode(self):
        self._check_finalized()
        if self._mode is None:
            if self._uniform_weights:
                counts = {}
                util.progress_bar_init('Computing mode...', self._length, 'Values')
                warnings.warn('Empirical has uniform weights and the mode will be correct only if values in Empirical are hashable.')
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
                _, max_index = util.to_tensor(self.log_weights).max(-1)
                self._mode = self._get_value(int(max_index))
        return self._mode

    @property
    def median(self):
        self._check_finalized()
        if self._median is None:
            if self._uniform_weights:
                if torch.is_tensor(self.values[0]):
                    values = torch.stack(self.get_values())
                    self._median = torch.median(values)
                else:
                    values = self.values_numpy()
                    self._median = np.median(values)
            else:
                # Resample to get an unweighted distribution
                self._median = self.resample(1000).median
        return self._median

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
        ret = Empirical(values=self.get_values(), name=self.name, *args, **kwargs)
        ret._metadata = copy.deepcopy(self._metadata)
        ret.add_metadata(op='discard_weights')
        return ret

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

    def density_estimate(self, num_mixture_components=1, num_samples=1000, *args, **kwargs):
        if self.weighted:
            values = util.to_tensor([self.sample() for _ in range(num_samples)]).cpu().numpy()
        else:
            values = self.values_numpy()
        if values.ndim != 1:
            raise ValueError('Expecting values to be one-dimensional')
        values = values.reshape(-1, 1)
        m = mixture.GaussianMixture(n_components=num_mixture_components, covariance_type='diag', *args, **kwargs)
        m.fit(values)
        dists = [Normal(m.means_[i][0], np.sqrt(m.covariances_[i])[0]) for i in range(num_mixture_components)]
        weights = [m.weights_[i] for i in range(num_mixture_components)]
        return Mixture(dists, weights)

    def combine_duplicates(self, *args, **kwargs):
        self._check_finalized()
        if self._type == EmpiricalType.MEMORY:
            distribution = collections.defaultdict(float)
            # This can be simplified once PyTorch supports content-based hashing of tensors. See: https://github.com/pytorch/pytorch/issues/2569
            hashable = util.is_hashable(self.values[0])
            if hashable:
                for i in range(self.length):
                    found = False
                    for key, value in distribution.items():
                        if torch.equal(util.to_tensor(key), util.to_tensor(self.values[i])):
                            # Differentiability warning: values[i] is discarded here. If we need to differentiate through all values, the gradients of values[i] and key should be tied here.
                            distribution[key] = torch.logsumexp(torch.stack((value, self.log_weights[i])), dim=0)
                            found = True
                    if not found:
                        distribution[self.values[i]] = self.log_weights[i]
                values = list(distribution.keys())
                log_weights = list(distribution.values())
                ret = Empirical(values=values, log_weights=log_weights, name=self.name, *args, **kwargs)
                ret._metadata = copy.deepcopy(self._metadata)
                ret.add_metadata(op='combine_duplicates')
                return ret
            else:
                raise RuntimeError('The values in this Empirical as not hashable. Combining of duplicates not currently supported.')
        else:
            raise NotImplementedError('Not implemented for type: {}'.format(str(self._type)))

    # Deprecated in favor of concat_empiricals in constructor
    # @staticmethod
    # def combine(empirical_distributions, file_name=None):
    #     empirical_type = empirical_distributions[0]._type
    #     for dist in empirical_distributions:
    #         if dist._type != empirical_type:
    #             raise RuntimeError('Expecting all Empirical distributions to be of the same type. Encountered: {} and {}'.format(empirical_type, dist._type))
    #         if not isinstance(dist, Empirical):
    #             raise TypeError('Combination is only supported between Empirical distributions.')
    #
    #     if empirical_type == EmpiricalType.FILE:
    #         if file_name is None:
    #             raise RuntimeError('Expecting a target file_name for the combined Empirical.')
    #         ret = Empirical(file_name=file_name)
    #         for dist in empirical_distributions:
    #             for i in range(dist._length):
    #                 ret.add(value=dist._shelf[str(i)], log_weight=dist._log_weights[i])
    #         ret.finalize()
    #         return ret
    #     elif empirical_type == EmpiricalType.MEMORY:
    #         values = []
    #         log_weights = []
    #         length = empirical_distributions[0].length
    #         for dist in empirical_distributions:
    #             if dist.length != length:
    #                 raise RuntimeError('Combination is only supported between Empirical distributions of equal length.')
    #             values += dist._values
    #             log_weights += dist._log_weights
    #         return Empirical(values=values, log_weights=log_weights, file_name=file_name)
    #     else:
    #         raise NotImplementedError('Not implemented for type: {}'.format(str(empirical_type)))

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

    def plot(self, *args, **kwargs):
        self.plot_histogram(*args, **kwargs)

    def plot_histogram(self, figsize=(10, 5), xlabel=None, ylabel='Frequency', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, density=1, fig=None, *args, **kwargs):
        if fig is None:
            if not show:
                mpl.rcParams['axes.unicode_minus'] = False
                plt.switch_backend('agg')
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
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
        if file_name is not None:
            plt.savefig(file_name)
        if show:
            plt.show()

    def save_metadata(self, file_name):
        with open(file_name, 'w') as file:
            file.write(yaml.dump(self._metadata))

    def __repr__(self):
        return 'Empirical(items:{}, weighted:{})'.format(self.length, self.weighted)
