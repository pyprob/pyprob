import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torch.distributed as dist
import math
import os
import sys
import h5py
import ujson
import time
import hashlib
import bisect
from glob import glob
import numpy as np
import uuid
from termcolor import colored
from collections import Counter, OrderedDict
import random
from pathlib import Path
from distutils.util import strtobool
import multiprocessing

from .. import util
from .utils import MyFile, construct_distribution as construct_dist
from ..util import TraceMode, PriorInflation
from ..concurrency import ConcurrentShelf
from ..trace import Trace, Variable


class Batch():
    def __init__(self, traces_and_hashes):
        self.traces_lists, hashes = zip(*traces_and_hashes)

        sub_batches = {}
        total_length = 0
        np_traces = np.asarray(self.traces_lists)
        np_hashes = np.asarray(hashes)
        uniques = np.unique(np_hashes)
        self.size = len(self.traces_lists)
        total_length_controlled = 0

        batch_splitting = [np_traces[np_hashes==h] for h in uniques]
        self.sub_batches = [[]]*len(uniques)

        for i, sub_batch_traces in enumerate(batch_splitting):

            example_trace = sub_batch_traces[0]
            trace_len = len(example_trace)

            values = []
            dist_parameters = []
            torch_data = []

            for _ in range(trace_len):
                values.append([])
                dist_parameters.append({})
                torch_data.append({})

            names = []
            controls = []
            dist_names = []
            observed_time_steps = []
            latent_time_steps = []
            addresses = []
            constants = []
            accepted = []
            reused = []
            tagged = []

            meta_data = {}
            meta_data['trace_hash'] = uniques[i]

            tl = 0
            for time_step, var_args in enumerate(example_trace):
                name = var_args['name']
                names.append(name)
                if var_args['observed']:
                    observed_time_steps.append(time_step)
                else:
                    latent_time_steps.append(time_step)

                dist_names.append(var_args['distribution_name'])
                addresses.append(var_args['address'])
                controls.append(var_args['control'])
                constants.append(var_args['constants'])
                accepted.append(var_args['accepted'])
                reused.append(var_args['reused'])
                tagged.append(var_args['tagged'])

                if var_args['control']:
                    tl += 1

            total_length_controlled += tl

            meta_data['names'] = names
            meta_data['observed_time_steps'] = observed_time_steps
            meta_data['latent_time_steps'] = latent_time_steps
            meta_data['distribution_names'] = dist_names
            meta_data['addresses'] = addresses
            meta_data['controls'] = controls
            meta_data['distribution_constants'] = constants
            meta_data['accepted'] = accepted
            meta_data['reused'] = reused
            meta_data['tagged'] = tagged

            for time_step in range(trace_len):
                for trace_list in sub_batch_traces:
                    var_args = trace_list[time_step]
                    value = var_args['value']
                    values[time_step].append(value)

                    for k, v in var_args['distribution_args'].items():
                        d = dist_parameters[time_step]

                        # add batch dimension
                        v = v.unsqueeze(0)
                        if k not in d:
                            d[k] = v
                        else:
                            d[k] = torch.cat([d[k], v], dim=0)

                torch_data[time_step]['values'] = torch.stack(values[time_step],
                                                              dim=0)
                torch_data[time_step]['distribution'] = construct_dist(dist_names[time_step],
                                                                       dist_parameters[time_step])

            self.sub_batches[i] = [meta_data, torch_data]
        self.mean_length_controlled = total_length_controlled / self.size

    def __len__(self):
        return len(self.traces_lists)

    # TODO!
    # def to(self, device):
    #     """ Sends data onto the desired device

    #     NOT DONE

    #     """
    #     for sub_batch in self.sub_batches:
    #         data = sub_batch[1]
    #         for d_time_step in data:
    #             for k, v in d_time_step.items():
    #                 v.to(device=device)

    # TODO!
    #def pin_memory(self):
    #    pass


class OnlineDataset(Dataset):
    def __init__(self, model, length=None,
                 prior_inflation=PriorInflation.DISABLED,
                 variables_observed_inf_training=[]):

        self._variables_observed_inf_training = variables_observed_inf_training

        self._model = model
        if length is None:
            length = int(1e6)
        self._length = length
        self._prior_inflation = prior_inflation

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK,
                                                  prior_inflation=self._prior_inflation))

        trace_attr_list = []
        for variable in trace.variables:
            trace_attr_list.append(util.to_variable_dict_data(variable,
                                                              self._variables_observed_inf_training, to_list=False))
        return trace_attr_list, trace.hash()

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file, for_inference=True, *args, **kwargs):
        num_files = math.ceil(num_traces / num_traces_per_file)
        util.progress_bar_init('Saving offline dataset, traces:{}, traces per file:{}, files:{}'.format(num_traces, num_traces_per_file, num_files), num_traces, 'Traces')
        i = 0
        str_type = h5py.special_dtype(vlen=str)
        while i < num_traces:
            file_name = os.path.join(dataset_dir, 'pyprob_traces_{}_{}'.format(num_traces_per_file, str(uuid.uuid4())))
            if for_inference:
                file_name = f'{file_name}_pruned'
            dataset = []
            hashes = []
            with h5py.File(file_name+".hdf5", 'w') as f:
                for j in range(num_traces_per_file):
                    if for_inference:
                        trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK,
                                                              prior_inflation=self._prior_inflation,
                                                              *args, **kwargs))
                    else:
                        trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR,
                                                                prior_inflation=self._prior_inflation,
                                                                *args, **kwargs))
                    trace_attr_list = []
                    for variable in trace.variables:
                        trace_attr_list.append(util.to_variable_dict_data(variable))

                    # call trace.__hash__ method for hashing
                    trace_hash = trace.hash()
                    hashes.append(trace_hash)
                    dataset.append(ujson.dumps([trace_attr_list, trace_hash]).encode())
                    util.progress_bar_update(i+j)

                f.create_dataset('traces', (num_traces_per_file,), data=dataset,
                                 chunks=True, dtype=str_type)

                f.attrs['num_traces'] = num_traces_per_file
                f.attrs['hashes'] = hashes
            i += num_traces_per_file
        util.progress_bar_end()

    def get_example_trace(self):
        trace_list, trace_hash = self[0]
        trace = Trace(trace_hash=trace_hash)
        for var_args in trace_list:
            distribution = construct_dist(var_args['distribution_name'], var_args['distribution_args'])
            variable = Variable(distribution=distribution, **var_args)
            trace.add(variable)
        trace.end(None, None)
        return trace

class OfflineDatasetFile(Dataset):

    def __init__(self, file_name, variables_observed_inf_training):
        self._variables_observed_inf_training = variables_observed_inf_training
        self._file_name = str(file_name.resolve())
        with h5py.File(self._file_name, 'r') as f:
            self._length = f.attrs['num_traces']
            self.hashes = f.attrs['hashes']

        # BELOW WE OPEN FILE HANDLERS USING THE SPECIAL MyFile - this may be a better/faster option

        #self.f = h5py.File(MyFile(str(self._file_name.resolve())), 'r')['traces']

    def __len__(self):
        return int(self._length)

    def __getitem__(self, idx):

        with h5py.File(self._file_name, 'r') as f:
            trace_attr_list, trace_hash = ujson.loads(f['traces'][idx])

        trace_list = util.from_variable_dict_data(trace_attr_list,
                                                  self._variables_observed_inf_training)
        return trace_list, trace_hash

    def get_example_trace(self):
        trace_list, trace_hash = self[0]
        trace = Trace(trace_hash=trace_hash)
        for var_args in trace_list:
            distribution = construct_dist(var_args['distribution_name'], var_args['distribution_args'])
            variable = Variable(distribution=distribution, **var_args)
            trace.add(variable)
        return trace

class OfflineDataset(ConcatDataset):
    def __init__(self, dataset_dir, variables_observed_inf_training=[]):
        p = Path(dataset_dir)
        assert(p.is_dir())
        files = sorted(p.glob('pyprob_traces*.hdf5'))
        if len(files) == 0:
            raise RuntimeError('Cannot find any data set files at {}'.format(dataset_dir))
        self.datasets = []
        for file_name in files:
            try:
                dataset = OfflineDatasetFile(file_name, variables_observed_inf_training)
                self.datasets.append(dataset)
            except Exception as e:
                print(e)
                print(colored('Warning: dataset file potentially corrupt, omitting: {}'.format(file_name), 'red', attrs=['bold']))
        super().__init__(self.datasets)
        print('OfflineDataset at: {}'.format(dataset_dir))
        print('Num. traces      : {:,}'.format(len(self)))
        hashes = [h for dataset in self.datasets
                    for h in dataset.hashes]
        self._hashes = hashes
        print(colored('Sorting'))
        self._sorted_indices = np.argsort(hashes, kind='mergesort')
        print(colored('Finished sorting hashes'))

    def get_example_trace(self):
        return self.datasets[0].get_example_trace()

    def save_sorted(self, sorted_dataset_dir, num_traces_per_file=None, num_files=None, begin_file_index=None, end_file_index=None):
        if num_traces_per_file is not None:
            if num_files is not None:
                raise ValueError('Expecting either num_traces_per_file or num_files')
        else:
            if num_files is None:
                raise ValueError('Expecting either num_traces_per_file or num_files')
            else:
                num_traces_per_file = math.ceil(len(self) / num_files)

        if os.path.exists(sorted_dataset_dir):
            if len(glob(os.path.join(sorted_dataset_dir, '*'))) > 0:
                print(colored('Warning: target directory is not empty: {})'.format(sorted_dataset_dir), 'red', attrs=['bold']))
        util.create_path(sorted_dataset_dir, directory=True)
        file_indices = list(util.chunks(list(self._sorted_indices), num_traces_per_file))
        num_traces = len(self)
        num_files = len(file_indices)
        num_files_digits = len(str(num_files))
        file_name_template = 'pyprob_traces_sorted_{{:d}}_{{:0{}d}}'.format(num_files_digits)
        file_names = list(map(lambda x: os.path.join(sorted_dataset_dir, file_name_template.format(num_traces_per_file, x)), range(num_files)))
        if begin_file_index is None:
            begin_file_index = 0
        if end_file_index is None:
            end_file_index = num_files
        if begin_file_index < 0 or begin_file_index > end_file_index or end_file_index > num_files or end_file_index < begin_file_index:
            raise ValueError('Invalid indexes begin_file_index:{} and end_file_index: {}'.format(begin_file_index, end_file_index))

        print('Sorted offline dataset, traces: {}, traces per file: {}, files: {} (overall)'.format(num_traces, num_traces_per_file, num_files))
        util.progress_bar_init('Saving sorted files with indices in range [{}, {}) ({} of {} files overall)'.format(begin_file_index, end_file_index, end_file_index - begin_file_index, num_files), end_file_index - begin_file_index + 1, 'Files')
        j = 0
        str_type = h5py.special_dtype(vlen=str)
        for i in range(begin_file_index, end_file_index):
            j += 1
            file_name = file_names[i]
            print(file_name)

            with h5py.File(file_name+".hdf5", 'w') as f:
                dataset = []
                hashes = []
                for old_i in file_indices[i]:
                    trace_attr_list, trace_hash = self._get_hdf5_item(old_i)
                    dataset.append(ujson.dumps([trace_attr_list, trace_hash]).encode())
                    hashes.append(trace_hash)
                f.create_dataset('traces', (num_traces_per_file,), data=dataset,
                                chunks=True, dtype=str_type)
                f.attrs['num_traces'] = len(dataset)
                f.attrs['hashes'] = hashes

                util.progress_bar_update(j)
        util.progress_bar_end()

    def _get_hdf5_item(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        dataset = self.datasets[dataset_idx]
        with h5py.File(dataset._file_name, 'r') as f:
            trace_attr_list, trace_hash = ujson.loads(f['traces'][sample_idx])
        return trace_attr_list, trace_hash

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
