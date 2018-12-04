import torch
import os
import sys
import math
import shutil
import uuid
import tempfile
import tarfile
import random
from queue import Queue
from threading import Thread
from termcolor import colored

from .. import __version__, util, TraceMode


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

    def __getitem__(self, key):
        return self.traces[key]

    def to(self, device):
        for trace in self.traces:
            trace.to(device=device)


class BatchGeneratorOnline():
    def __init__(self, model, prior_inflation, batch_size):
        self._model = model
        self._prior_inflation = prior_inflation
        self._batch_size = batch_size

    def batches(self, pre_load_next=False):
        traces = self._model._traces(self._batch_size, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, silent=True).get_values()
        yield Batch(traces)

    def save_traces(self, trace_dir, num_traces=16, num_files=1, *args, **kwargs):
        num_traces_per_file = math.ceil(num_traces / num_files)
        util.progress_bar_init('Saving traces to disk, num_traces:{}, num_files:{}, num_traces_per_file:{}'.format(num_traces, num_files, num_traces_per_file), num_traces, 'Traces')
        i = 0
        while i < num_traces:
            i += num_traces_per_file
            traces = self._model._traces(num_traces=num_traces_per_file, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, silent=True, *args, **kwargs)
            file_name = os.path.join(trace_dir, 'pyprob_traces_{}_{}'.format(num_traces_per_file, str(uuid.uuid4())))
            self._save_traces(traces, file_name)
            util.progress_bar_update(i)
        util.progress_bar_end()

    def _save_traces(self, traces, file_name):
        data = {}
        data['traces'] = traces
        data['model_name'] = self._model.name
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_traces')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_traces')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()


class BatchGeneratorOffline():
    def __init__(self, trace_dir, batch_size):
        self._trace_dir = trace_dir
        self._files = self._get_files()
        self._batch_size = batch_size
        self._length = int(len(self._files) / self._batch_size)
        print('Using offline training traces with {} files at {}'.format(len(self._files), self._trace_dir))

    def __len__(self):
        return self._length

    def batches(self, pre_load_next=False):
        epoch_indices = list(range(len(self._files)))
        random.shuffle(epoch_indices)  # Happens once per epoch due to yield generator

        def _load_batch(indices):
            return Batch([self._load_trace(self._files[i]) for i in indices])

        if pre_load_next:
            queue = Queue()
            batch_indices = epoch_indices[:self._batch_size]
            epoch_indices[:self._batch_size] = []
            pre_loader = Thread(target=lambda q, i: q.put(_load_batch(i)), args=(queue, batch_indices))
            pre_loader.start()

        while len(epoch_indices) >= self._batch_size:
            batch_indices = epoch_indices[:self._batch_size]
            epoch_indices[:self._batch_size] = []
            if pre_load_next:
                pre_loader.join()
                batch = queue.get()
                pre_loader = Thread(target=lambda q, i: q.put(_load_batch(i)), args=(queue, batch_indices))
                pre_loader.start()
            else:
                batch = _load_batch(batch_indices)
            yield batch
        if pre_load_next:
            pre_loader.join()
            batch = queue.get()
            yield batch

    def _get_files(self):
        files = [name for name in os.listdir(self._trace_dir)]
        files = list(map(lambda f: os.path.join(self._trace_dir, f), files))
        return files

    def _load_trace(self, file_name):
        # try:
        tar = tarfile.open(file_name, 'r:gz')
        tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
        tmp_file = os.path.join(tmp_dir, 'pyprob_trace')
        tar.extract('pyprob_trace', tmp_dir)
        tar.close()
        if util._cuda_enabled:
            data = torch.load(tmp_file)
        else:
            data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
        shutil.rmtree(tmp_dir)
        # except:
        #     print(colored('Warning: cannot load traces from file, file potentially corrupt: {}'.format(file_name), 'red', attrs=['bold']))
        #     return []

        # print('Loading trace cache of size {}'.format(data['size']))
        # if data['model_name'] != self._model.name:
            # print(colored('Warning: different model names (loaded traces: {}, current model: {})'.format(data['model_name'], self._model.name), 'red', attrs=['bold']))
        # if data['pyprob_version'] != __version__:
        #     print(colored('Warning: different pyprob versions (loaded trace: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        # if data['torch_version'] != torch.__version__:
        #     print(colored('Warning: different PyTorch versions (loaded trace: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        trace = data['trace']
        trace.to(device=util._device)
        return trace
