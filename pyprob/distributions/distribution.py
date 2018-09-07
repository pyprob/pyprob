import torch
import tarfile
import tempfile
import shutil
import os
import uuid
import traceback
from threading import Thread
from termcolor import colored

from .. import __version__, util


class Distribution():
    def __init__(self, name, address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size(), torch_dist=None):
        self._name = name
        self._address_suffix = address_suffix
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._torch_dist = torch_dist

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(value)
            return lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        return torch.exp(self.log_prob(value))

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            return self._torch_dist.variance
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    def save(self, file_name):
        data = {}
        data['distribution'] = self
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_distribution')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_distribution')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def load(file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_distribution')
            tar.extract('pyprob_distribution', tmp_dir)
            tar.close()
            data = torch.load(tmp_file, map_location=torch.device('cpu'))
            shutil.rmtree(tmp_dir)
        except Exception as e:
            raise RuntimeError('Cannot load distribution. {}'.format(traceback.format_exc()))

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded distribution: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded distribution: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        return data['distribution']
