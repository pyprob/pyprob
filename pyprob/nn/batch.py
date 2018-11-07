import torch
import copy
import os
import shutil
import uuid
import tempfile
import tarfile
import random
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

    def to(self, device):
        for trace in self.traces:
            trace.to(device=device)


class BatchGeneratorOnline():
    def __init__(self, model, prior_inflation):
        self._model = model
        self._prior_inflation = prior_inflation

    def batches(self, size=64, discard_source=False, *args, **kwargs):
        traces = self._model._traces(size, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, silent=True, *args, **kwargs).get_values()
        yield Batch(traces)

    def save_traces(self, trace_dir, files=16, traces_per_file=16, *args, **kwargs):
        for file in range(files):
            traces = self._model._traces(traces_per_file, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, *args, **kwargs).get_values()
            file_name = os.path.join(trace_dir, 'pyprob_traces_{}_{}'.format(traces_per_file, str(uuid.uuid4())))
            self._save_traces(traces, file_name)
            print('Traces {:,}/{:,}, file {:,}/{:,}: {}'.format((file + 1) * traces_per_file, files * traces_per_file, file + 1, files, file_name))

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
    def __init__(self, trace_dir):
        self._trace_dir = trace_dir
        self._trace_files = self._trace_dir_files()
        self._trace_files_discarded = []
        print('Using offline training traces with {} files at {}'.format(len(self._trace_files), self._trace_dir))

    def batches(self, size=64, discard_source=False):
        random.shuffle(self._trace_files)  # Happens once per epoch due to yield generator
        trace_cache = []
        trace_files = copy.deepcopy(self._trace_files)
        while len(trace_files) > 0:
            while len(trace_cache) < size:
                file = trace_files.pop()
                new_traces = self._load_traces(file)
                trace_cache += new_traces
                if discard_source:
                    self._trace_files.remove(file)
                if len(trace_files) == 0:
                    break
            batch_traces = trace_cache[:size]
            trace_cache[:size] = []
            yield Batch(batch_traces)

    def _trace_dir_files(self):
        files = [name for name in os.listdir(self._trace_dir)]
        files = list(map(lambda f: os.path.join(self._trace_dir, f), files))
        return files

    def _load_traces(self, file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_traces')
            tar.extract('pyprob_traces', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
            shutil.rmtree(tmp_dir)
        except:
            print(colored('Warning: cannot load traces from file, file potentially corrupt: {}'.format(file_name), 'red', attrs=['bold']))
            return []

        # print('Loading trace cache of size {}'.format(data['size']))
        # if data['model_name'] != self._model.name:
            # print(colored('Warning: different model names (loaded traces: {}, current model: {})'.format(data['model_name'], self._model.name), 'red', attrs=['bold']))
        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded traces: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded traces: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        traces = data['traces']
        for trace in traces:
            trace.to(device=util._device)
        return traces
