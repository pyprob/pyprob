import torch
import time
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
        self.length = len(traces)
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


class BatchGenerator():
    def __init__(self, model, prior_inflation, trace_store_dir=None):
        self._model = model
        self._prior_inflation = prior_inflation
        self._trace_store_dir = trace_store_dir
        if trace_store_dir is not None:
            self._trace_store_cache = []
            self._trace_store_discarded_file_names = []
            num_files = len(self._trace_store_current_files())
            print('Monitoring trace cache (currently with {} files) at {}'.format(num_files, trace_store_dir))

    def get_batch(self, length=64, discard_source=False, *args, **kwargs):
        if self._trace_store_dir is None:
            # There is no trace store on disk, sample traces online from the model
            traces, _ = self._model._traces(length, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, silent=True, *args, **kwargs)
        else:
            # There is a trace store on disk, load traces from disk
            if discard_source:
                self._trace_store_cache = []

            while len(self._trace_store_cache) < length:
                current_files = self._trace_store_current_files()
                if len(current_files) == 0:
                    cache_is_empty = True
                    cache_was_empty = False
                    while cache_is_empty:
                        current_files = self._trace_store_current_files()
                        num_files = len(current_files)
                        if num_files > 0:
                            cache_is_empty = False
                            if cache_was_empty:
                                print('Resuming, new data appeared in trace cache (currently with {} files) at {}'.format(num_files, self._trace_cache_path))
                        else:
                            if not cache_was_empty:
                                print('Waiting for new data, empty (or fully discarded) trace cache at {}'.format(self._trace_cache_path))
                                cache_was_empty = True
                            time.sleep(0.5)

                current_file = random.choice(current_files)
                if discard_source:
                    self._trace_store_discarded_file_names.append(current_file)
                new_traces = self._load_traces(current_file)
                if len(new_traces) == 0:  # When empty or corrupt file is read
                    self._trace_store_discarded_file_names.append(current_file)
                else:
                    random.shuffle(new_traces)
                    self._trace_store_cache += new_traces

            traces = self._trace_store_cache[0:length]
            self._trace_store_cache[0:length] = []
        return Batch(traces)

    def save_trace_store(self, trace_store_dir, files=16, traces_per_file=16, *args, **kwargs):
        f = 0
        done = False
        while not done:
            traces, _ = self._model._traces(traces_per_file, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, *args, **kwargs)
            file_name = os.path.join(trace_store_dir, 'pyprob_traces_{}_{}'.format(traces_per_file, str(uuid.uuid4())))
            self._save_traces(traces, file_name)
            f += 1
            if (files is not None) and (f >= files):
                done = True

    def _trace_store_current_files(self):
        files = [name for name in os.listdir(self._trace_store_dir)]
        files = list(map(lambda f: os.path.join(self._trace_store_dir, f), files))
        for discarded_file_name in self._trace_store_discarded_file_names:
            if discarded_file_name in files:
                files.remove(discarded_file_name)
        return files

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

        # print('Loading trace cache of length {}'.format(data['length']))
        if data['model_name'] != self._model.name:
            print(colored('Warning: different model names (loaded traces: {}, current model: {})'.format(data['model_name'], self._model.name), 'red', attrs=['bold']))
        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded traces: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded traces: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        traces = data['traces']
        for trace in traces:
            trace.to(device=util._device)
        return traces
