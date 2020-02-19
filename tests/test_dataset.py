import unittest
import math
import torch
import tempfile
import shutil
from glob import glob
import os

import pyprob
from pyprob import util, Model
from pyprob.distributions import Normal, Uniform
from pyprob.nn import OfflineDataset
import pyprob.diagnostics


class DatasetTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMeanMarsaglia(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean (Marsaglia)')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform)
                    y = pyprob.sample(uniform)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, name='obs0')
                pyprob.observe(likelihood, name='obs1')
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_save_offline_dataset(self):
        dataset_dir = tempfile.mkdtemp()
        num_traces_correct = 20
        num_traces_per_file_correct = 5
        num_files_correct = 4

        self._model.save_dataset(dataset_dir=dataset_dir, num_traces=num_traces_correct, num_traces_per_file=num_traces_per_file_correct)
        files = sorted(glob(os.path.join(dataset_dir, 'pyprob_traces_*')))
        num_files = len(files)
        dataset = OfflineDataset(dataset_dir)
        hashes = dataset._hashes
        indices = dataset._sorted_indices
        sorted_on_disk = util.is_sorted(indices)
        num_traces = len(dataset)
        num_traces_per_file = num_traces / num_files
        shutil.rmtree(dataset_dir)

        util.eval_print('dataset_dir', 'num_traces', 'num_traces_correct', 'num_traces_per_file', 'num_traces_per_file_correct', 'files', 'num_files', 'num_files_correct', 'hashes', 'indices', 'sorted_on_disk')

        self.assertEqual(num_files, num_files_correct)
        self.assertEqual(num_traces, num_traces_correct)
        self.assertEqual(num_traces_per_file, num_traces_per_file_correct)
        self.assertFalse(sorted_on_disk)

    def test_sort_offline_dataset(self):
        unsorted_dataset_dir = tempfile.mkdtemp()
        unsorted_num_traces = 20
        unsorted_num_traces_per_file = 5
        sorted_dataset_dir = tempfile.mkdtemp()
        sorted_num_traces_per_file_correct = 2
        sorted_num_files_correct = 10

        self._model.save_dataset(dataset_dir=unsorted_dataset_dir, num_traces=unsorted_num_traces, num_traces_per_file=unsorted_num_traces_per_file)
        unsorted_files = sorted(glob(os.path.join(unsorted_dataset_dir, '*')))
        unsorted_dataset = OfflineDataset(unsorted_dataset_dir)
        unsorted_hashes = unsorted_dataset._hashes
        unsorted_indices = unsorted_dataset._sorted_indices
        unsorted_dataset.save_sorted(sorted_dataset_dir=sorted_dataset_dir, num_traces_per_file=sorted_num_traces_per_file_correct)
        shutil.rmtree(unsorted_dataset_dir)

        sorted_dataset = OfflineDataset(sorted_dataset_dir)
        sorted_files = sorted(glob(os.path.join(sorted_dataset_dir, '*')))
        sorted_num_traces_per_file = len(sorted_dataset.datasets[0])
        sorted_num_files = len(sorted_dataset.datasets)
        sorted_num_traces = sorted_num_traces_per_file * sorted_num_files
        sorted_hashes = [trace_hash for _, trace_hash in sorted_dataset]
        sorted_indices = sorted_dataset._sorted_indices
        sorted_on_disk = util.is_sorted(sorted_indices)
        shutil.rmtree(sorted_dataset_dir)

        util.eval_print('unsorted_dataset_dir', 'unsorted_num_traces', 'unsorted_num_traces_per_file', 'unsorted_files', 'unsorted_hashes', 'unsorted_indices', 'sorted_dataset_dir', 'sorted_files', 'sorted_hashes', 'sorted_indices', 'sorted_on_disk', 'sorted_num_traces', 'sorted_num_files', 'sorted_num_files_correct', 'sorted_num_traces_per_file', 'sorted_num_traces_per_file_correct')

        self.assertTrue(sorted_on_disk)
        self.assertEqual(sorted_num_files, sorted_num_files_correct)
        self.assertEqual(sorted_num_traces, unsorted_num_traces)
        self.assertEqual(sorted_num_traces_per_file, sorted_num_traces_per_file_correct)

    def test_sort_offline_dataset_multi_node(self):
        unsorted_dataset_dir = tempfile.mkdtemp()
        unsorted_num_traces = 20
        unsorted_num_traces_per_file = 5
        sorted_dataset_dir = tempfile.mkdtemp()
        sorted_num_traces_per_file_correct = 2
        sorted_num_files_correct = 10

        self._model.save_dataset(dataset_dir=unsorted_dataset_dir, num_traces=unsorted_num_traces, num_traces_per_file=unsorted_num_traces_per_file)
        unsorted_files = sorted(glob(os.path.join(unsorted_dataset_dir, '*')))
        unsorted_dataset = OfflineDataset(unsorted_dataset_dir)
        unsorted_hashes = unsorted_dataset._hashes
        unsorted_indices = unsorted_dataset._sorted_indices
        unsorted_dataset.save_sorted(sorted_dataset_dir=sorted_dataset_dir, num_traces_per_file=sorted_num_traces_per_file_correct,                     end_file_index=4)
        unsorted_dataset.save_sorted(sorted_dataset_dir=sorted_dataset_dir, num_traces_per_file=sorted_num_traces_per_file_correct, begin_file_index=4, end_file_index=6)
        unsorted_dataset.save_sorted(sorted_dataset_dir=sorted_dataset_dir, num_traces_per_file=sorted_num_traces_per_file_correct, begin_file_index=6)
        shutil.rmtree(unsorted_dataset_dir)

        sorted_dataset = OfflineDataset(sorted_dataset_dir)
        sorted_files = sorted(glob(os.path.join(sorted_dataset_dir, '*')))
        sorted_num_traces_per_file = len(sorted_dataset.datasets[0])
        sorted_num_files = len(sorted_dataset.datasets)
        sorted_num_traces = sorted_num_traces_per_file * sorted_num_files
        sorted_hashes = [trace_hash for _, trace_hash in sorted_dataset]
        sorted_indices = sorted_dataset._sorted_indices
        sorted_on_disk = util.is_sorted(sorted_indices)
        shutil.rmtree(sorted_dataset_dir)

        util.eval_print('unsorted_dataset_dir', 'unsorted_num_traces', 'unsorted_num_traces_per_file', 'unsorted_files', 'unsorted_hashes', 'unsorted_indices', 'sorted_dataset_dir', 'sorted_files', 'sorted_hashes', 'sorted_indices', 'sorted_on_disk', 'sorted_num_traces', 'sorted_num_files', 'sorted_num_files_correct', 'sorted_num_traces_per_file', 'sorted_num_traces_per_file_correct')

        self.assertTrue(sorted_on_disk)
        self.assertEqual(sorted_num_files, sorted_num_files_correct)
        self.assertEqual(sorted_num_traces, unsorted_num_traces)
        self.assertEqual(sorted_num_traces_per_file, sorted_num_traces_per_file_correct)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
