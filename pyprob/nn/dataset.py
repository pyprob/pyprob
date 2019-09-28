import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torch.distributed as dist
import math
import os
import sys
import shelve
from glob import glob
import numpy as np
import uuid
from termcolor import colored
from collections import Counter, OrderedDict
import random

from .. import util
from ..util import TraceMode, PriorInflation
from ..concurrency import ConcurrentShelf


class Batch():
    def __init__(self, traces):
        self.traces = traces
        self.size = len(traces)
        sub_batches = {}
        total_length_controlled = 0
        for trace in traces:
            tl = trace.length_controlled
            if tl == 0:
                raise ValueError('Trace of length zero.')
            total_length_controlled += tl
            trace_hash = ''.join([variable.address for variable in trace.variables_controlled])
            if trace_hash not in sub_batches:
                sub_batches[trace_hash] = []
            sub_batches[trace_hash].append(trace)
        self.sub_batches = list(sub_batches.values())
        self.mean_length_controlled = total_length_controlled / self.size

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
        return next(self._model._trace_generator(trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK, prior_inflation=self._prior_inflation))

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
        # del(trace.log_importance_weight)
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
            controlled = False
            for v in trace.variables_controlled:
                if variable is v:  # Needs to be implemented this way to compare object references instead of object hashes (which change as a result of potentially deleted fields)
                    controlled = True
                    break
            if not controlled:
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
            shelf = shelve.open(file_name, flag='c')
            for j in range(num_traces_per_file):
                trace = next(self._model._trace_generator(trace_mode=TraceMode.PRIOR_FOR_INFERENCE_NETWORK, prior_inflation=self._prior_inflation, *args, **kwargs))
                self._prune_trace(trace)
                shelf[str(j)] = trace
                shelf['__length'] = j + 1
            shelf.close()
            util.progress_bar_update(i)
        util.progress_bar_end()


class OfflineDatasetFile(Dataset):
    cache = OrderedDict()
    cache_capacity = 8

    def __init__(self, file_name):
        self._file_name = file_name
        self._closed = False
        shelf = self._open()
        self._length = shelf['__length']

    def _open(self):
        # idea from https://www.kunxi.org/2014/05/lru-cache-in-python
        try:
            shelf = OfflineDatasetFile.cache.pop(self._file_name)
            # it was in the cache, put it back on the front
            OfflineDatasetFile.cache[self._file_name] = shelf
            return shelf
        except KeyError:
            # not in the cache
            if len(OfflineDatasetFile.cache) >= OfflineDatasetFile.cache_capacity:
                # cache is full, delete the last entry
                n, s = OfflineDatasetFile.cache.popitem(last=False)
                s.close()
            shelf = shelve.open(self._file_name, flag='r')
            OfflineDatasetFile.cache[self._file_name] = shelf
            return shelf

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        shelf = self._open()
        return shelf[str(idx)]


class OfflineDataset(ConcatDataset):
    def __init__(self, dataset_dir):
        self._dataset_dir = dataset_dir
        # files = [name for name in os.listdir(self._dataset_dir)]
        files = sorted(glob(os.path.join(self._dataset_dir, 'pyprob_traces_sorted_*')))
        if len(files) > 0:
            self._sorted_on_disk = True
        else:
            self._sorted_on_disk = False
            files = sorted(glob(os.path.join(self._dataset_dir, 'pyprob_traces_*')))
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
        print('OfflineDataset at: {}'.format(self._dataset_dir))
        print('Num. traces      : {:,}'.format(len(self)))
        print('Sorted on disk   : {}'.format(self._sorted_on_disk))
        if self._sorted_on_disk:
            self._sorted_indices = list(range(len(self)))
        else:
            file_name = os.path.join(self._dataset_dir, 'pyprob_hashes')
            try:
                hashes_file = shelve.open(file_name, 'r')
                hashes_exist = 'hashes' in hashes_file
                hashes_file.close()
            except:
                hashes_exist = False
            if hashes_exist:
                print('Using pre-computed hashes in: {}'.format(file_name))
                hashes_file = shelve.open(file_name, 'r')
                self._hashes = hashes_file['hashes']
                self._sorted_indices = hashes_file['sorted_indices']
                hashes_file.close()
                if torch.is_tensor(self._hashes):
                    self._hashes = self._hashes.cpu().numpy()
                if len(self._sorted_indices) != len(self):
                    raise RuntimeError('Length of pre-computed hashes ({}) and length of offline dataset ({}) do not match. Dataset files have been altered. Delete and re-generate pre-computed hash file: {}'.format(len(self._sorted_indices), len(self), file_name))
            else:
                print('No pre-computed hashes found, generating: {}'.format(file_name))
                hashes_file = shelve.open(file_name, 'c')
                hashes, sorted_indices = self._compute_hashes()
                hashes_file['hashes'] = hashes
                hashes_file['sorted_indices'] = sorted_indices
                hashes_file.close()
                self._sorted_indices = sorted_indices
                self._hashes = hashes
            print('Num. trace types : {:,}'.format(len(set(self._hashes))))
            hashes_and_counts = OrderedDict(sorted(Counter(self._hashes).items()))
            print('Trace hash\tCount')
            for hash, count in hashes_and_counts.items():
                print('{:.8f}\t{}'.format(hash, count))
        print()

    @staticmethod
    def _trace_hash(trace):
        h = hash(''.join([variable.address for variable in trace.variables_controlled])) + sys.maxsize + 1
        return float('{}.{}'.format(trace.length_controlled, h))

    def _compute_hashes(self):
        hashes = torch.zeros(len(self))
        util.progress_bar_init('Hashing offline dataset for sorting', len(self), 'Traces')
        for i in range(len(self)):
            hashes[i] = self._trace_hash(self[i])
            util.progress_bar_update(i)
        util.progress_bar_end()
        print('Sorting offline dataset')
        _, sorted_indices = torch.sort(hashes)
        print('Sorting done')
        return hashes.cpu().numpy(), sorted_indices.cpu().numpy()

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
        for i in range(begin_file_index, end_file_index):
            j += 1
            file_name = file_names[i]
            print(file_name)
            shelf = ConcurrentShelf(file_name)
            shelf.lock(write=True)
            for new_i, old_i in enumerate(file_indices[i]):
                shelf[str(new_i)] = self[old_i]
            shelf['__length'] = len(file_indices[i])
            shelf.unlock()
            util.progress_bar_update(j)
        util.progress_bar_end()


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
