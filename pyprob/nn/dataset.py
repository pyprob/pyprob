import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import math
import os
import sys
from glob import glob
import numpy as np
import uuid
from termcolor import colored

from .. import util
from ..util import TraceMode, PriorInflation
from ..concurrency import ConcurrentShelf


class Batch():
    def __init__(self, traces):
        self.traces = traces
        self.size = len(traces)
        sub_batches = {}
        for trace in traces:
            if trace.length == 0:
                raise ValueError('Trace of length zero.')
            trace_hash = ''.join([variable.address for variable in trace.variables_controlled])
            if trace_hash not in sub_batches:
                sub_batches[trace_hash] = []
            sub_batches[trace_hash].append(trace)
        self.sub_batches = list(sub_batches.values())

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, key):
        return self.traces[key]

    def to(self, device):
        for trace in self.traces:
            trace.to(device=device)


class OnlineDataset(Dataset):
    def __init__(self, model, length=None, prior_inflation=PriorInflation.DISABLED):
        self._model = model
        if length is None:
            length = int(1e6)
        self._length = length
        self._prior_inflation = prior_inflation

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return next(self._model._trace_generator(trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation))

    @staticmethod
    def _prune_trace(trace):
        del(trace.variables)
        # trace.variables_controlled = []
        del(trace.variables_uncontrolled)
        del(trace.variables_replaced)
        del(trace.variables_observed)
        del(trace.variables_observable)
        del(trace.variables_tagged)
        del(trace.variables_dict_address)
        del(trace.variables_dict_address_base)
        # trace.named_variables = {}
        del(trace.result)
        del(trace.log_prob)
        del(trace.log_prob_observed)
        del(trace.log_importance_weight)
        # trace.length = 0
        # trace.length_controlled = 0
        del(trace.execution_time_sec)
        for variable in trace.variables_controlled:
            # variable.distribution = distribution
            # if value is None:
            #     variable.value = None
            # else:
            #     variable.value = util.to_tensor(value)
            del(variable.address_base)
            # variable.address = address
            del(variable.instance)
            del(variable.log_prob)
            del(variable.control)
            del(variable.replace)
            del(variable.name)
            del(variable.observable)
            del(variable.observed)
            del(variable.reused)
            del(variable.tagged)
        for _, variable in trace.named_variables.items():
            del(variable.distribution)
            # if value is None:
            #     variable.value = None
            # else:
            #     variable.value = util.to_tensor(value)
            del(variable.address_base)
            del(variable.address)
            del(variable.instance)
            del(variable.log_prob)
            del(variable.control)
            del(variable.replace)
            del(variable.name)
            del(variable.observable)
            del(variable.observed)
            del(variable.reused)
            del(variable.tagged)

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file, *args, **kwargs):
        num_files = math.ceil(num_traces / num_traces_per_file)
        util.progress_bar_init('Saving offline dataset, traces:{}, traces per file:{}, files:{}'.format(num_traces, num_traces_per_file, num_files), num_traces, 'Traces')
        i = 0
        while i < num_traces:
            i += num_traces_per_file
            file_name = os.path.join(dataset_dir, 'pyprob_traces_{}_{}'.format(num_traces_per_file, str(uuid.uuid4())))
            shelf = ConcurrentShelf(file_name)
            shelf.lock(write=True)
            for j in range(num_traces_per_file):
                trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, *args, **kwargs))
                self._prune_trace(trace)
                shelf[str(j)] = trace
                shelf['__length'] = j + 1
            shelf.unlock()
            util.progress_bar_update(i)
        util.progress_bar_end()


class OfflineDatasetFile(Dataset):
    def __init__(self, file_name):
        self._file_name = file_name
        self._shelf = ConcurrentShelf(file_name)
        self._length = self._shelf['__length']

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self._shelf[str(idx)]


class OfflineDataset(ConcatDataset):
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir
        # files = [name for name in os.listdir(self._dataset_dir)]
        files = sorted(glob(os.path.join(self._dataset_dir, 'pyprob_traces_*')))
        datasets = []
        for file in files:
            try:
                dataset = OfflineDatasetFile(file)
                datasets.append(dataset)
            except Exception as e:
                print(e)
                print(colored('Warning: dataset file potentially corrupt, omitting: {}'.format(file), 'red', attrs=['bold']))
        super().__init__(datasets)
        file_name = os.path.join(self._dataset_dir, 'pyprob_hashes')
        hashes_file = ConcurrentShelf(file_name)
        if 'hashes' in hashes_file:
            print('Using pre-computed hashes in: {}'.format(file_name))
            self._sorted_indices = hashes_file['sorted_indices']
        else:
            print('No pre-computed hashes found, generating: {}'.format(file_name))
            hashes, sorted_indices = self._compute_hashes()
            hashes_file['hashes'] = hashes
            hashes_file['sorted_indices'] = sorted_indices
            self._sorted_indices = sorted_indices

    def _compute_hashes(self):
        def trace_hash(trace):
            h = hash(''.join([variable.address for variable in trace.variables_controlled])) + sys.maxsize + 1
            return float('{}.{}'.format(trace.length_controlled, h))
        hashes = torch.zeros(len(self))
        util.progress_bar_init('Hashing offline dataset for sorting', len(self), 'Traces')
        for i in range(len(self)):
            hashes[i] = trace_hash(self[i])
            util.progress_bar_update(i)
        util.progress_bar_end()
        print('Sorting offline dataset')
        _, sorted_indices = torch.sort(hashes)
        print('Sorting done')
        return hashes, sorted_indices.cpu().numpy()


class SortedTraceSampler(Sampler):
    def __init__(self, offline_dataset):
        self._sorted_indices = offline_dataset._sorted_indices

    def __iter__(self):
        return iter(self._sorted_indices)

    def __len__(self):
        return len(self._offline_dataset)
