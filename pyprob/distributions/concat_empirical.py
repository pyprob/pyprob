import torch
import numpy as np
import math

from . import Distribution, Empirical
from .. import util


class ConcatEmpirical(Distribution):
    def __init__(self, empirical_dists=None, empirical_file_names=None):
        if empirical_dists is not None:
            if type(empirical_dists) == list:
                if type(empirical_dists[0]) == Empirical:
                    self._empiricals = empirical_dists
                else:
                    raise TypeError('Expecting empirical_dists to be a list of Empiricals.')
            else:
                raise TypeError('Expecting empirical_dists to be a list of Empiricals.')
        else:
            if empirical_file_names is not None:
                if type(empirical_file_names) == list:
                    if type(empirical_file_names[0]) == str:
                        self._empiricals = [Empirical(file_name=f, file_read_only=True) for f in empirical_file_names]
                    else:
                        raise TypeError('Expecting empirical_file_names to be a list of file names.')
                else:
                    raise TypeError('Expecting empirical_file_names to be a list of file names.')
            else:
                raise ValueError('Expecting either empirical_dists or empirical_file_names.')
        self._cum_sizes = np.cumsum([emp.length for emp in self._empiricals])
        self._length = self._cum_sizes[-1]
        self._log_weights = torch.cat([util.to_tensor(emp._log_weights) for emp in self._empiricals])
        self._categorical = torch.distributions.Categorical(logits=util.to_tensor(self._log_weights, dtype=torch.float64))
        weights = self._categorical.probs
        self._effective_sample_size = 1. / weights.pow(2).sum()
        super().__init__('Combined empirical, traces: {:,}, ESS: {:,.2f}'.format(self._length, self._effective_sample_size))

    def _get_value(self, index):
        emp_index = self._cum_sizes.searchsorted(index, 'right')
        if emp_index > 0:
            index = index - self._cum_sizes[emp_index - 1]
        return self._empiricals[emp_index]._get_value(index)

    def _get_log_weight(self, index):
        return self._categorical.logits[index]

    def _get_weight(self, index):
        return self._categorical.probs[index]

    def sample(self):
        index = int(self._categorical.sample())
        return self._get_value(index)

    def resample(self, num_samples, map_func=None):
        if map_func is None:
            map_func = lambda x: x
        values = []
        util.progress_bar_init('Resampling', num_samples, 'Samples')
        for i in range(num_samples):
            util.progress_bar_update(i)
            values.append(map_func(self.sample()))
        util.progress_bar_end()
        return Empirical(values=values, name=self.name)

    def thin(self, num_samples, map_func=None, min_index=None, max_index=None):
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
        return Empirical(values=values, log_weights=log_weights, name=self.name)
