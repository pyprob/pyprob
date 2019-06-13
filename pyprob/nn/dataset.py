import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torch.distributed as dist
import math
import os
import sys
import h5py
import json
import time
import hashlib
from glob import glob
import numpy as np
import uuid
from termcolor import colored
from collections import Counter, OrderedDict
import random
from pathlib import Path
from distutils.util import strtobool

from .. import util
from ..util import TraceMode, PriorInflation
from ..concurrency import ConcurrentShelf
from ..trace import Trace, Variable

VARIABLE_ATTRIBUTES = ['value', 'address_base', 'address', 'instance', 'log_prob',
                       'control', 'tagged', 'constants', 'observed', 'name',
                       'distribution_name', 'distribution_args']

class Batch():
    def __init__(self, traces_and_hashes):
        self.traces, hashes = zip(*traces_and_hashes)
        self.num_sub_traces = len(self.traces)
        sub_batches = {}
        total_length = 0
        np_traces = np.asarray(self.traces)
        np_hashes = np.asarray(hashes)

        self.sub_batches = [np_traces[np_hashes==h] for h in np.unique(np_hashes)]

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, key):
        return self.traces[key]

    def to(self, device):
        for trace in self.traces:
            trace.to(device=device)


class OnlineDataset(Dataset):
    def __init__(self, model, length=None,
                 prior_inflation=PriorInflation.DISABLED):
        self._model = model
        if length is None:
            length = int(1e6)
        self._length = length
        self._prior_inflation = prior_inflation

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        trace =next(self._model._trace_generator(trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK,
                                                 prior_inflation=self._prior_inflation))
        return trace, trace.hash()

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file, *args, **kwargs):
        num_files = math.ceil(num_traces / num_traces_per_file)
        util.progress_bar_init('Saving offline dataset, traces:{}, traces per file:{}, files:{}'.format(num_traces, num_traces_per_file, num_files), num_traces, 'Traces')
        i = 0
        str_type = h5py.special_dtype(vlen=str)
        while i < num_traces:
            i += num_traces_per_file
            file_name = os.path.join(dataset_dir, 'pyprob_traces_{}_{}'.format(num_traces_per_file, str(uuid.uuid4())))
            dataset = {}
            hashes = []
            with h5py.File(file_name+".hdf5", 'w') as f:
                for j in range(num_traces_per_file):
                    trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR,
                                                              prior_inflation=self._prior_inflation,
                                                              *args, **kwargs))

                    # call trace.__hash__ method for hashing
                    trace_hash = trace.hash()

                    trace_len = trace.length

                    if trace_hash not in f.keys():
                        hashes.append(trace_hash)
                        grp = f.create_group(trace_hash)
                        dataset[trace_hash] = {}
                        dataset[trace_hash]['trace_len'] = trace_len
                        dataset[trace_hash]['traces'] = []

                    tmp = []

                    for attr in VARIABLE_ATTRIBUTES:
                        attr_list = self._update_sub_trace_data([], attr, trace)
                        tmp.append(attr_list)

                    dataset[trace_hash]['traces'].append(tmp)

                for trace_hash, sub_trace_dataset in dataset.items():
                    grp = f[trace_hash]
                    trace_len = sub_trace_dataset['trace_len']
                    num_sub_traces = len(sub_trace_dataset['traces'])
                    grp.attrs['num_sub_traces'] = num_sub_traces
                    grp.create_dataset('traces', (num_sub_traces, len(VARIABLE_ATTRIBUTES), trace_len),
                                       data=sub_trace_dataset['traces'],
                                       chunks=(min(int(num_sub_traces), 100), len(VARIABLE_ATTRIBUTES),
                                               trace_len),
                                       compression="gzip", dtype=str_type)

                f.attrs['num_traces'] = num_traces_per_file
                f.attrs['hashes'] = hashes
            util.progress_bar_update(i)
        util.progress_bar_end()

    def _update_sub_trace_data(self, sub_trace_data, attr, trace):
        """ Update sub_trace_data dictionary

        Inputs:

        sub_trace_data -- list of list, so changes will be "by reference" and are done outside this function
        attr           -- str - attribute
        trace          -- trace object for which we loop over it variables

        We further decode the json string into bytes (which has to be decoded once we load the data again)

        """
        for variable in trace.variables:
            if attr == 'value':
                v = json.dumps(getattr(variable, attr).unsqueeze(0).tolist()).encode()
            elif attr in ['distribution_name']:
                # extract the input arguments for initializing the distribution
                v = json.dumps(variable.distribution_name).encode()
            elif attr in ['distribution_args']:
                v = json.dumps(variable.distribution_args).encode()
            elif attr in ['constants']:
                tmp = {}
                for k, value in variable.constants.items():
                    tmp[k] = value.tolist()
                v = json.dumps(tmp).encode()
            elif attr in ['log_prob']:
                v = json.dumps(getattr(variable, attr).item()).encode()
            else:
                v = json.dumps(getattr(variable, attr)).encode()
            sub_trace_data.append(v)

        return sub_trace_data


class OfflineDatasetFile(Dataset):

    data_cache = {}
    # specifies the number of file we have open at a time
    cache_size = 100

    def __init__(self, file_name):
        from ..state import _variables_observed_inf_training
        self._variables_observed_inf_training = _variables_observed_inf_training
        self._file_name = file_name
        with h5py.File(file_name, 'r') as f:
            self._length = f.attrs['num_traces']
            self.hashes = f.attrs['hashes']
            self.n_traces_per_trace = [f[h].attrs['num_sub_traces'] for h in self.hashes]

    def __len__(self):
        return int(self._length)

    def __getitem__(self, idx):
        trace_attributes, trace_hash = self.get_data(idx)

        trace = Trace(trace_hash=trace_hash)
        n_variables = len(trace_attributes[0])
        var_args = []
        for _ in range(n_variables):
            var_args.append({})

        for attr, variable_attributes in zip(VARIABLE_ATTRIBUTES, trace_attributes):
            for t, variable_data in enumerate(variable_attributes):
                if attr == 'value':
                    v = util.to_tensor(json.loads(variable_data))
                elif attr in ['distribution_name']:
                    # extract the input arguments for initializing the distribution
                    v = json.loads(variable_data)
                elif attr in ['distribution_args']:
                    v = json.loads(variable_data)
                elif attr in ['constants']:
                    tmp = {}
                    constants = json.loads(variable_data)
                    for k, value in constants.items():
                        tmp[k] = util.to_tensor(value)
                    v = tmp
                elif attr in ['log_prob']:
                    v = util.to_tensor(json.loads(variable_data))
                elif attr in ['reused', 'tagged', 'control', 'observed']:
                    v = json.loads(variable_data)
                else:
                    v = json.loads(variable_data)

                var_args[t][attr] = v

        for args in var_args:
            args['observed'] = args['observed'] or args['name'] in self._variables_observed_inf_training
            variable = Variable(**args)
            trace.add(variable)

        trace.end(None, None)

        return trace, trace_hash

    def _load_data(self):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        # remove an element from data cache if size will be exceeded
        if len(self.data_cache) == self.cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            self.data_cache.pop(removal_keys[0])

        with h5py.File(self._file_name) as f:
            cum_index = 0
            for trace_hash in self.hashes:
                sub_traces = f[trace_hash]
                # add data to the data cache and retrieve
                # the cache index
                self._add_to_cache(sub_traces['traces'], trace_hash, self._file_name)

    @classmethod
    def _add_to_cache(cls, sub_traces, trace_hash, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        for data in sub_traces:
            data_entry = [data, trace_hash]
            if file_path not in cls.data_cache:
                cls.data_cache[file_path] = [data_entry]
            else:
                cls.data_cache[file_path].append(data_entry)

    def get_data(self, idx):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """

        if self._file_name not in self.data_cache:
            self._load_data()

        # extract data from cache
        return self.data_cache[self._file_name][idx]


class OfflineDataset(ConcatDataset):
    def __init__(self, dataset_dir):
        p = Path(dataset_dir)
        assert(p.is_dir())
        files = sorted(p.glob('pyprob_traces*.hdf5'))
        if len(files) == 0:
            raise RuntimeError('Cannot find any data set files at {}'.format(dataset_dir))
        datasets = []
        for file in files:
            try:
                dataset = OfflineDatasetFile(file)
                datasets.append(dataset)
            except Exception as e:
                print(e)
                print(colored('Warning: dataset file potentially corrupt, omitting: {}'.format(file), 'red', attrs=['bold']))
        super().__init__(datasets)
        print('OfflineDataset at: {}'.format(dataset_dir))
        print('Num. traces      : {:,}'.format(len(self)))
        hashes = [h*n_traces_per_hash for dataset in datasets
                                      for h, n_traces_per_hash in zip(dataset.hashes,
                                                                      dataset.n_traces_per_trace)]
        print(colored('Sorting'))
        self._sorted_indices = np.argsort(hashes)
        print(colored('Finished sorting hashes'))


class TraceSampler(Sampler):
    def __init__(self, offline_dataset):
        if not isinstance(offline_dataset, OfflineDataset):
            raise TypeError('Expecting an OfflineDataset instance.')
        self._sorted_indices = offline_dataset._sorted_indices

    def __iter__(self):
        return iter(self._sorted_indices)

    def __len__(self):
        return len(self._offline_dataset)


class TraceBatchSampler(Sampler):
    def __init__(self, offline_dataset, batch_size, shuffle_batches=True):
        if not isinstance(offline_dataset, OfflineDataset):
            raise TypeError('Expecting an OfflineDataset instance.')
        self._batches = list(util.chunks(offline_dataset._sorted_indices, batch_size))
        self._shuffle_batches = shuffle_batches

    def __iter__(self):
        if self._shuffle_batches:
            np.random.shuffle(self._batches)
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class DistributedTraceBatchSampler(Sampler):
    def __init__(self, offline_dataset, batch_size, shuffle_batches=True, num_buckets=None, shuffle_buckets=True):
        if not isinstance(offline_dataset, OfflineDataset):
            raise TypeError('Expecting an OfflineDataset instance.')
        if not dist.is_available():
            raise RuntimeError('Expecting distributed training.')
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        # Randomly drop a number of traces so that the number of all minibatches in the whole dataset is an integer multiple of world size
        num_batches_to_drop = math.floor(len(offline_dataset._sorted_indices) / batch_size) % self._world_size
        num_traces_to_drop = num_batches_to_drop * batch_size
        # Ensure all ranks choose the same traces to drop
        st = random.getstate()
        random.seed(0)
        self._batches = list(util.chunks(util.drop_items(list(offline_dataset._sorted_indices), num_traces_to_drop), batch_size)) # List of all minibatches, where each minibatch is a list of trace indices
        random.setstate(st)
        # Discard last minibatch if it's smaller than batch_size
        if len(self._batches[-1]) < batch_size:
            del(self._batches[-1])
        if num_buckets is None:
            num_buckets = len(self._batches) / self._world_size
        self._num_buckets = num_buckets
        self._bucket_size = math.ceil(len(self._batches) / num_buckets)
        if self._bucket_size < self._world_size:
            raise RuntimeError('offline_dataset:{}, batch_size:{} and num_buckets:{} imply a bucket_size:{} smaller than world_size:{}'.format(len(offline_dataset), batch_size, num_buckets, self._bucket_size, self._world_size))
        # List of buckets, where each bucket is a list of minibatches
        self._buckets = list(util.chunks(self._batches, self._bucket_size))
        # Unify last two buckets if the last bucket is smaller than other buckets
        if len(self._buckets[-1]) < self._bucket_size:
            if len(self._buckets) < 2:
                raise RuntimeError('offline_dataset:{} too small for given batch_size:{} and num_buckets:{}'.format(len(offline_dataset), batch_size, num_buckets))
            self._buckets[-2].extend(self._buckets[-1])
            del(self._buckets[-1])
        self._shuffle_batches = shuffle_batches
        self._shuffle_buckets = shuffle_buckets
        self._epoch = 0
        self._current_bucket_id = 0

        print('DistributedTraceBatchSampler')
        print('OfflineDataset size : {:,}'.format(len(offline_dataset)))
        print('World size          : {:,}'.format(self._world_size))
        print('Batch size          : {:,}'.format(batch_size))
        print('Num. batches dropped: {:,}'.format(num_batches_to_drop))
        print('Num. batches        : {:,}'.format(len(self._batches)))
        print('Bucket size         : {:,}'.format(self._bucket_size))
        print('Num. buckets        : {:,}'.format(self._num_buckets))

    def __iter__(self):
        self._epoch += 1
        bucket_ids = list(range(len(self._buckets)))
        if self._shuffle_buckets:
            # Shuffle the list of buckets (but not the order of minibatches inside each bucket) at the beginning of each epoch, deterministically based on the epoch number so that all nodes have the same bucket order
            # Idea from: https://github.com/pytorch/pytorch/blob/a3fb004b1829880547dd7b3e2cd9d16af657b869/torch/utils/data/distributed.py#L44
            st = np.random.get_state()
            np.random.seed(self._epoch)
            np.random.shuffle(bucket_ids)
            np.random.set_state(st)
        for bucket_id in bucket_ids:
            bucket = self._buckets[bucket_id]
            self._current_bucket_id = bucket_id
            # num_batches is needed to ensure that all nodes have the same number of minibatches (iterations) in each bucket, in cases where the bucket size is not divisible by world_size.
            num_batches = math.floor(len(bucket) / self._world_size)
            # Select a num_batches-sized subset of the current bucket for the current node
            # The part not selected by the current node will be selected by other nodes
            batches = bucket[self._rank:len(bucket):self._world_size][:num_batches]
            if self._shuffle_batches:
                # Shuffle the list of minibatches (but not the order trace indices inside each minibatch) selected for the current node
                np.random.shuffle(batches)
            for batch in batches:
                yield batch

    def __len__(self):
        return len(self._batches)
