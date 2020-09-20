import unittest
import torch
import numpy as np
import os
import math
import uuid
import tempfile

import pyprob
from pyprob import util, Model
from pyprob.distributions import Empirical, Normal, Categorical, Uniform, Poisson, Beta, Bernoulli, Exponential, Gamma, LogNormal, Binomial, Weibull, VonMises, Mixture, TruncatedNormal, Factor


empirical_samples = 25000


class GaussianWithUnknownMean(Model):
    def __init__(self):
        super().__init__('Gaussian with unknown mean')

    def forward(self):
        mu = pyprob.sample(Normal(1, math.sqrt(5)), name='mu')
        likelihood = Normal(mu, math.sqrt(2))
        pyprob.observe(likelihood, name='obs0')
        pyprob.observe(likelihood, name='obs1')
        return mu


class DistributionsTestCase(unittest.TestCase):
    def test_distributions_empirical(self):
        values = util.to_tensor([1, 2, 3])
        log_weights = util.to_tensor([1, 2, 3])
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675
        dist_expectation_sin_correct = 0.3921678960323334
        dist_map_sin_mean_correct = 0.3921678960323334
        dist_min_correct = 1
        dist_max_correct = 3
        dist_mode_correct = 3
        dist_unweighted_mean_correct = 2
        dist_unweighted_stddev_correct = 0.816497

        dist = Empirical(values, log_weights)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_expectation_sin = float(dist.expectation(torch.sin))
        dist_map_sin_mean = float(dist.map(torch.sin).mean)
        dist_min = float(dist.min)
        dist_max = float(dist.max)
        dist_mode = float(dist.mode)
        dist_unweighted = dist.unweighted()
        dist_unweighted_mean = float(dist_unweighted.mean)
        dist_unweighted_stddev = float(dist_unweighted.stddev)

        util.eval_print('dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct', 'dist_min', 'dist_min_correct', 'dist_max', 'dist_max_correct', 'dist_mode', 'dist_mode_correct', 'dist_unweighted_mean', 'dist_unweighted_mean_correct', 'dist_unweighted_stddev', 'dist_unweighted_stddev_correct')

        # self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_expectation_sin, dist_expectation_sin_correct, places=1)
        self.assertAlmostEqual(dist_map_sin_mean, dist_map_sin_mean_correct, places=1)
        self.assertAlmostEqual(dist_min, dist_min_correct, places=1)
        self.assertAlmostEqual(dist_max, dist_max_correct, places=1)
        self.assertAlmostEqual(dist_mode, dist_mode_correct, places=1)
        self.assertAlmostEqual(dist_unweighted_mean, dist_unweighted_mean_correct, places=1)
        self.assertAlmostEqual(dist_unweighted_stddev, dist_unweighted_stddev_correct, places=1)

    def test_distributions_empirical_copy(self):
        file_name_1 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        values = util.to_tensor([1, 2, 3])
        log_weights = util.to_tensor([1, 2, 3])
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675

        dist_1 = Empirical(values, log_weights)  # In memory
        dist_1_mean = float(dist_1.mean)
        dist_1_stddev = float(dist_1.stddev)

        dist_2 = dist_1.copy()  # In memory
        dist_2_mean = float(dist_2.mean)
        dist_2_stddev = float(dist_2.stddev)

        dist_3 = dist_2.copy(file_name=file_name_1)  # On disk
        dist_3_mean = float(dist_3.mean)
        dist_3_stddev = float(dist_3.stddev)

        dist_4 = dist_3.copy(file_name=file_name_2)  # On disk
        dist_4_mean = float(dist_4.mean)
        dist_4_stddev = float(dist_4.stddev)

        dist_5 = dist_4.copy()  # In memory
        dist_5_mean = float(dist_5.mean)
        dist_5_stddev = float(dist_5.stddev)

        util.eval_print('dist_1_mean', 'dist_2_mean', 'dist_3_mean', 'dist_4_mean', 'dist_5_mean', 'dist_mean_correct', 'dist_1_stddev', 'dist_2_stddev', 'dist_3_stddev', 'dist_4_stddev', 'dist_5_stddev', 'dist_stddev_correct')

        self.assertAlmostEqual(dist_1_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_1_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_2_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_2_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_3_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_3_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_4_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_4_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_5_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_5_stddev, dist_stddev_correct, places=1)

    def test_distributions_empirical_disk(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        values = util.to_tensor([1, 2, 3])
        log_weights = util.to_tensor([1, 2, 3])
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675
        dist_expectation_sin_correct = 0.3921678960323334
        dist_map_sin_mean_correct = 0.3921678960323334
        dist_min_correct = 1
        dist_max_correct = 3
        dist_mode_correct = 3
        dist_unweighted_mean_correct = 2
        dist_unweighted_stddev_correct = 0.816497

        dist = Empirical(values=values, log_weights=log_weights, file_name=file_name)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_expectation_sin = float(dist.expectation(torch.sin))
        dist_map_sin_mean = float(dist.map(torch.sin).mean)
        dist_min = float(dist.min)
        dist_max = float(dist.max)
        dist_mode = float(dist.mode)
        dist_unweighted = dist.copy().unweighted()
        dist_unweighted_mean = float(dist_unweighted.mean)
        dist_unweighted_stddev = float(dist_unweighted.stddev)

        util.eval_print('dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct', 'dist_min', 'dist_min_correct', 'dist_max', 'dist_max_correct', 'dist_mode', 'dist_mode_correct', 'dist_unweighted_mean', 'dist_unweighted_mean_correct', 'dist_unweighted_stddev', 'dist_unweighted_stddev_correct')

        # self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_expectation_sin, dist_expectation_sin_correct, places=1)
        self.assertAlmostEqual(dist_map_sin_mean, dist_map_sin_mean_correct, places=1)
        self.assertAlmostEqual(dist_min, dist_min_correct, places=1)
        self.assertAlmostEqual(dist_max, dist_max_correct, places=1)
        self.assertAlmostEqual(dist_mode, dist_mode_correct, places=1)
        self.assertAlmostEqual(dist_unweighted_mean, dist_unweighted_mean_correct, places=1)
        self.assertAlmostEqual(dist_unweighted_stddev, dist_unweighted_stddev_correct, places=1)

    def test_distributions_empirical_disk_append(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        dist_means_correct = -1.2
        dist_stddevs_correct = 1.4
        dist_empirical_length_correct = 2000

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical(file_name=file_name)
        dist_empirical.add_sequence([dist.sample() for i in range(1000)])
        dist_empirical.finalize()
        dist_empirical.close()
        dist_empirical_2 = Empirical(file_name=file_name)
        dist_empirical_2.add_sequence([dist.sample() for i in range(1000)])
        dist_empirical_2.finalize()
        dist_empirical_length = dist_empirical_2.length
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical_2.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical_2.stddev)

        util.eval_print('dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_empirical_length', 'dist_empirical_length_correct')

        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertEqual(dist_empirical_length, dist_empirical_length_correct)

    def test_distributions_empirical_combine_duplicates(self):
        values = [1, 2, 2, 3, 3, 3]
        values_combined_correct = [1, 2, 3]
        dist_mean_correct = 2.333333
        dist_stddev_correct = 0.745356

        dist = Empirical(values)
        dist_combined = dist.combine_duplicates()
        values_combined = dist_combined.get_values()

        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)
        dist_mean_combined = float(dist_combined.mean)
        dist_stddev_combined = float(dist_combined.stddev)

        util.eval_print('values', 'values_combined', 'values_combined_correct', 'dist_mean', 'dist_mean_combined', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_combined', 'dist_stddev_correct')
        self.assertEqual(set(values_combined), set(values_combined_correct))
        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_combined, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_combined, dist_stddev_correct, places=1)

    def test_distributions_empirical_density_estimate(self):
        values = [1, 2, 3]
        dist_mixture_means_correct = [1, 2, 3]

        dist = Empirical(values)
        dist_mixture = dist.density_estimate(num_mixture_components=3)
        dist_mixture_means = [float(d.mean) for d in dist_mixture.distributions]

        util.eval_print('values', 'dist_mixture_means', 'dist_mixture_means_correct')

        self.assertAlmostEqual(set(dist_mixture_means), set(dist_mixture_means_correct), places=1)

    def test_distributions_empirical_numpy(self):
        samples = 25
        dist_means_correct = 10
        dist_stddevs_correct = 0.01

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(samples)])
        dist_empirical_values_numpy = dist_empirical.values_numpy()
        dist_empirical_values_numpy_len = len(dist_empirical_values_numpy)
        dist_empirical_values_numpy_mean = np.mean(dist_empirical_values_numpy)
        dist_empirical_values_numpy_stddev = np.std(dist_empirical_values_numpy)
        dist_empirical_weights_numpy = dist_empirical.weights_numpy()
        dist_empirical_weights_numpy_len = len(dist_empirical_weights_numpy)

        util.eval_print('samples', 'dist_empirical_values_numpy_len', 'dist_empirical_weights_numpy_len', 'dist_empirical_values_numpy_mean', 'dist_means_correct', 'dist_empirical_values_numpy_stddev', 'dist_stddevs_correct')

        self.assertEqual(dist_empirical_values_numpy_len, samples)
        self.assertEqual(dist_empirical_weights_numpy_len, samples)
        self.assertAlmostEqual(dist_empirical_values_numpy_mean, dist_means_correct, places=1)
        self.assertAlmostEqual(dist_empirical_values_numpy_stddev, dist_stddevs_correct, places=0)

    def test_distributions_empirical_resample(self):
        dist_means_correct = [2]
        dist_stddevs_correct = [5]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_empirical = dist_empirical.resample(int(empirical_samples/2))
        dist_metadata = dist_empirical.metadata
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        util.eval_print('dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_metadata')

        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))


    def test_distributions_empirical_resample_disk(self):
        dist_means_correct = [2]
        dist_stddevs_correct = [5]
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_empirical = dist_empirical.resample(int(empirical_samples/8))
        dist_empirical.copy(file_name=file_name)
        dist_empirical_disk = Empirical(file_name=file_name)
        dist_metadata = dist_empirical_disk.metadata
        dist_means_empirical = util.to_numpy(dist_empirical_disk.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical_disk.stddev)

        util.eval_print('file_name', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_metadata')

        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))

    def test_distributions_empirical_thin(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        dist_thinned_values_correct = [1, 4, 7, 10]

        dist = Empirical(values)
        dist_thinned = dist.thin(4)
        dist_thinned_values = list(dist_thinned.values_numpy())

        util.eval_print('dist_thinned_values', 'dist_thinned_values_correct')

        self.assertEqual(dist_thinned_values, dist_thinned_values_correct)

    def test_distributions_empirical_thin_disk(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        dist_thinned_values_correct = [1, 4, 7, 10]
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        dist = Empirical(values)
        dist_thinned = dist.thin(4)
        dist_thinned.copy(file_name=file_name)
        dist_thinned_disk = Empirical(file_name=file_name)
        dist_thinned_values = list(dist_thinned_disk.values_numpy())

        util.eval_print('file_name', 'dist_thinned_values', 'dist_thinned_values_correct')

        self.assertEqual(dist_thinned_values, dist_thinned_values_correct)

    def test_distributions_empirical_slice_and_index(self):
        dist_slice_elements_correct = [0, 1, 2]
        dist_first_correct = 0
        dist_last_correct = 5

        dist = Empirical([0, 1, 2, 3, 4, 5])
        dist_slice_elements = dist[0:3].get_values()
        dist_first = dist[0]
        dist_last = dist[-1]

        util.eval_print('dist_slice_elements', 'dist_slice_elements_correct', 'dist_first', 'dist_first_correct', 'dist_last', 'dist_last_correct')

        self.assertEqual(dist_slice_elements, dist_slice_elements_correct)
        self.assertEqual(dist_first, dist_first_correct)
        self.assertEqual(dist_last, dist_last_correct)

    def test_distributions_empirical_slice_and_index_disk(self):
        dist_slice_elements_correct = [0, 1, 2]
        dist_first_correct = 0
        dist_last_correct = 5
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        dist = Empirical([0, 1, 2, 3, 4, 5], file_name=file_name)
        dist_slice_elements = dist[0:3].get_values()
        dist_first = dist[0]
        dist_last = dist[-1]

        util.eval_print('file_name', 'dist_slice_elements', 'dist_slice_elements_correct', 'dist_first', 'dist_first_correct', 'dist_last', 'dist_last_correct')

        self.assertEqual(dist_slice_elements, dist_slice_elements_correct)
        self.assertEqual(dist_first, dist_first_correct)
        self.assertEqual(dist_last, dist_last_correct)

    def test_distributions_empirical_sample_min_max_index(self):
        dist_mean_1_correct = 2
        dist_mean_2_correct = 3
        dist_mean_3_correct = 4
        #                 0  1  2  3  4  5  6  7  8
        dist = Empirical([2, 2, 2, 2, 3, 3, 3, 4, 4])
        dist_mean_1 = float(Empirical([dist.sample(min_index=0, max_index=3)]).mean)
        dist_mean_2 = float(Empirical([dist.sample(min_index=4, max_index=6)]).mean)
        dist_mean_3 = float(Empirical([dist.sample(min_index=7, max_index=8)]).mean)

        util.eval_print('dist_mean_1', 'dist_mean_1_correct', 'dist_mean_2', 'dist_mean_2_correct', 'dist_mean_3', 'dist_mean_3_correct')

        self.assertAlmostEqual(dist_mean_1, dist_mean_1_correct, places=1)
        self.assertAlmostEqual(dist_mean_2, dist_mean_2_correct, places=1)
        self.assertAlmostEqual(dist_mean_3, dist_mean_3_correct, places=1)

    def test_distributions_empirical_combine_unweighted(self):
        dist1_mean_correct = 1
        dist1_stddev_correct = 3
        dist2_mean_correct = 5
        dist2_stddev_correct = 2
        dist3_mean_correct = -2.5
        dist3_stddev_correct = 1.2
        dist_combined_mean_correct = 1.16667
        dist_combined_stddev_correct = 3.76858
        dist_combined_weighted_correct = False

        empirical_samples = 100000
        dist1 = Normal(dist1_mean_correct, dist1_stddev_correct)
        dist1_empirical = Empirical([dist1.sample() for i in range(empirical_samples)])
        dist1_mean_empirical = float(dist1_empirical.mean)
        dist1_stddev_empirical = float(dist1_empirical.stddev)

        dist2 = Normal(dist2_mean_correct, dist2_stddev_correct)
        dist2_empirical = Empirical([dist2.sample() for i in range(empirical_samples)])
        dist2_mean_empirical = float(dist2_empirical.mean)
        dist2_stddev_empirical = float(dist2_empirical.stddev)

        dist3 = Normal(dist3_mean_correct, dist3_stddev_correct)
        dist3_empirical = Empirical([dist3.sample() for i in range(empirical_samples)])
        dist3_mean_empirical = float(dist3_empirical.mean)
        dist3_stddev_empirical = float(dist3_empirical.stddev)

        dist_combined_empirical = Empirical(concat_empiricals=[dist1_empirical, dist2_empirical, dist3_empirical])
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)
        dist_combined_weighted = dist_combined_empirical.weighted

        util.eval_print('dist1_mean_empirical', 'dist1_stddev_empirical', 'dist1_mean_correct', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_stddev_empirical', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_stddev_empirical', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_stddev_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_correct', 'dist_combined_weighted', 'dist_combined_weighted_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=1)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=1)
        self.assertEqual(dist_combined_weighted, dist_combined_weighted_correct)

    def test_distributions_empirical_disk_combine_unweighted(self):
        file_name_1 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_3 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_combined = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        dist1_mean_correct = 1
        dist1_stddev_correct = 3
        dist2_mean_correct = 5
        dist2_stddev_correct = 2
        dist3_mean_correct = -2.5
        dist3_stddev_correct = 1.2
        dist_combined_mean_correct = 1.16667
        dist_combined_stddev_correct = 3.76858
        dist_combined_weighted_correct = False

        dist1 = Normal(dist1_mean_correct, dist1_stddev_correct)
        dist1_empirical = Empirical([dist1.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_1)
        dist1_mean_empirical = float(dist1_empirical.mean)
        dist1_stddev_empirical = float(dist1_empirical.stddev)
        dist1_empirical.close()

        dist2 = Normal(dist2_mean_correct, dist2_stddev_correct)
        dist2_empirical = Empirical([dist2.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_2)
        dist2_mean_empirical = float(dist2_empirical.mean)
        dist2_stddev_empirical = float(dist2_empirical.stddev)
        dist2_empirical.close()

        dist3 = Normal(dist3_mean_correct, dist3_stddev_correct)
        dist3_empirical = Empirical([dist3.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_3)
        dist3_mean_empirical = float(dist3_empirical.mean)
        dist3_stddev_empirical = float(dist3_empirical.stddev)
        dist3_empirical.close()

        dist_combined_empirical = Empirical(concat_empirical_file_names=[file_name_1, file_name_2, file_name_3], file_name=file_name_combined)
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)
        dist_combined_weighted = dist_combined_empirical.weighted

        util.eval_print('dist1_mean_empirical', 'dist1_mean_correct', 'dist1_stddev_empirical', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_mean_correct', 'dist2_stddev_empirical', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_mean_correct', 'dist3_stddev_empirical', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_empirical', 'dist_combined_stddev_correct', 'dist_combined_weighted', 'dist_combined_weighted_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=0)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=0)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=0)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=0)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=0)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=0)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=0)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=0)
        self.assertEqual(dist_combined_weighted, dist_combined_weighted_correct)

    def test_distributions_empirical_combine_weighted(self):
        dist1_values = [1, 2, 3]
        dist1_log_weights = [1, 2, 3]
        dist1_mean_correct = 2.5752103328704834
        dist1_stddev_correct = 0.6514633893966675
        dist2_values = [1.4, -9, 5]
        dist2_log_weights = [-10, -2, -3]
        dist2_mean_correct = -5.233193397521973
        dist2_stddev_correct = 6.207840442657471
        dist3_values = [10, 4, -1]
        dist3_log_weights = [1, -2, -2.5]
        dist3_mean_correct = 9.415830612182617
        dist3_stddev_correct = 2.168320417404175
        dist_combined_mean_correct = 3.1346240043640137
        dist_combined_stddev_correct = 2.2721681594848633
        dist_combined_weighted_correct = True

        dist1_empirical = Empirical(values=dist1_values, log_weights=dist1_log_weights)
        dist1_mean_empirical = float(dist1_empirical.mean)
        dist1_stddev_empirical = float(dist1_empirical.stddev)

        dist2_empirical = Empirical(values=dist2_values, log_weights=dist2_log_weights)
        dist2_mean_empirical = float(dist2_empirical.mean)
        dist2_stddev_empirical = float(dist2_empirical.stddev)

        dist3_empirical = Empirical(values=dist3_values, log_weights=dist3_log_weights)
        dist3_mean_empirical = float(dist3_empirical.mean)
        dist3_stddev_empirical = float(dist3_empirical.stddev)

        dist_combined_empirical = Empirical(concat_empiricals=[dist1_empirical, dist2_empirical, dist3_empirical])
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)
        dist_combined_weighted = dist_combined_empirical.weighted

        util.eval_print('dist1_mean_empirical', 'dist1_mean_correct', 'dist1_stddev_empirical', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_mean_correct', 'dist2_stddev_empirical', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_mean_correct', 'dist3_stddev_empirical', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_empirical', 'dist_combined_stddev_correct', 'dist_combined_weighted', 'dist_combined_weighted_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=1)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=1)
        self.assertEqual(dist_combined_weighted, dist_combined_weighted_correct)

    def test_distributions_empirical_disk_combine_weighted(self):
        file_name_1 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_3 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_combined = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        dist1_values = [1, 2, 3]
        dist1_log_weights = [1, 2, 3]
        dist1_mean_correct = 2.5752103328704834
        dist1_stddev_correct = 0.6514633893966675
        dist2_values = [1.4, -9, 5]
        dist2_log_weights = [-10, -2, -3]
        dist2_mean_correct = -5.233193397521973
        dist2_stddev_correct = 6.207840442657471
        dist3_values = [10, 4, -1]
        dist3_log_weights = [1, -2, -2.5]
        dist3_mean_correct = 9.415830612182617
        dist3_stddev_correct = 2.168320417404175
        dist_combined_mean_correct = 3.1346240043640137
        dist_combined_stddev_correct = 2.2721681594848633
        dist_combined_weighted_correct = True

        dist1_empirical = Empirical(values=dist1_values, log_weights=dist1_log_weights, file_name=file_name_1)
        dist1_mean_empirical = float(dist1_empirical.mean)
        dist1_stddev_empirical = float(dist1_empirical.stddev)
        dist1_empirical.close()

        dist2_empirical = Empirical(values=dist2_values, log_weights=dist2_log_weights, file_name=file_name_2)
        dist2_mean_empirical = float(dist2_empirical.mean)
        dist2_stddev_empirical = float(dist2_empirical.stddev)
        dist2_empirical.close()

        dist3_empirical = Empirical(values=dist3_values, log_weights=dist3_log_weights, file_name=file_name_3)
        dist3_mean_empirical = float(dist3_empirical.mean)
        dist3_stddev_empirical = float(dist3_empirical.stddev)
        dist3_empirical.close()

        dist_combined_empirical = Empirical(concat_empirical_file_names=[file_name_1, file_name_2, file_name_3], file_name=file_name_combined)
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)
        dist_combined_weighted = dist_combined_empirical.weighted

        util.eval_print('dist1_mean_empirical', 'dist1_mean_correct', 'dist1_stddev_empirical', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_mean_correct', 'dist2_stddev_empirical', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_mean_correct', 'dist3_stddev_empirical', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_empirical', 'dist_combined_stddev_correct', 'dist_combined_weighted', 'dist_combined_weighted_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=0)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=0)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=0)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=0)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=0)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=0)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=0)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=0)
        self.assertEqual(dist_combined_weighted, dist_combined_weighted_correct)

    def test_distributions_empirical_save_load(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        values = util.to_tensor([1, 2, 3])
        log_weights = util.to_tensor([1, 2, 3])
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675
        dist_expectation_sin_correct = 0.3921678960323334
        dist_map_sin_mean_correct = 0.3921678960323334

        dist_on_file = Empirical(values, log_weights=log_weights, file_name=file_name)
        dist_on_file.close()
        dist = Empirical(file_name=file_name)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_expectation_sin = float(dist.expectation(torch.sin))
        dist_map_sin_mean = float(dist.map(torch.sin).mean)
        os.remove(file_name)

        util.eval_print('file_name', 'dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_expectation_sin, dist_expectation_sin_correct, places=1)
        self.assertAlmostEqual(dist_map_sin_mean, dist_map_sin_mean_correct, places=1)

    def test_distributions_empirical_concat_mem_to_mem(self):
        values_correct = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
        log_weights_correct = [-10, -15, -200, -2, -3, -22, -100, 1, 2, -0.3]
        mean_correct = 7.741360664367676
        stddev_correct = 0.7910336256027222
        ess_correct = 1.9459790014029552

        dist_correct = Empirical(values=values_correct, log_weights=log_weights_correct)
        dist_correct_mean = float(dist_correct.mean)
        dist_correct_stddev = float(dist_correct.stddev)
        dist_correct_ess = float(dist_correct.effective_sample_size)
        empiricals = []
        empiricals.append(Empirical(values=values_correct[0:3], log_weights=log_weights_correct[0:3]))
        empiricals.append(Empirical(values=values_correct[3:5], log_weights=log_weights_correct[3:5]))
        empiricals.append(Empirical(values=values_correct[5:9], log_weights=log_weights_correct[5:9]))
        empiricals.append(Empirical(values=values_correct[9:10], log_weights=log_weights_correct[9:10]))
        concat_emp = Empirical(concat_empiricals=empiricals)
        concat_emp_mean = float(concat_emp.mean)
        concat_emp_stddev = float(concat_emp.stddev)
        concat_emp_ess = float(concat_emp.effective_sample_size)

        util.eval_print('values_correct', 'log_weights_correct', 'dist_correct_mean', 'concat_emp_mean', 'mean_correct', 'dist_correct_stddev', 'concat_emp_stddev', 'stddev_correct', 'dist_correct_ess', 'concat_emp_ess', 'ess_correct')

        self.assertAlmostEqual(dist_correct_mean, mean_correct, places=1)
        self.assertAlmostEqual(dist_correct_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(dist_correct_ess, ess_correct, places=1)
        self.assertAlmostEqual(concat_emp_mean, mean_correct, places=1)
        self.assertAlmostEqual(concat_emp_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(concat_emp_ess, ess_correct, places=1)

    def test_distributions_empirical_concat_file_to_mem(self):
        values_correct = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
        log_weights_correct = [-10, -15, -200, -2, -3, -22, -100, 1, 2, -0.3]
        mean_correct = 7.741360664367676
        stddev_correct = 0.7910336256027222
        ess_correct = 1.9459790014029552

        dist_correct = Empirical(values=values_correct, log_weights=log_weights_correct)
        dist_correct_mean = float(dist_correct.mean)
        dist_correct_stddev = float(dist_correct.stddev)
        dist_correct_ess = float(dist_correct.effective_sample_size)
        file_names = [os.path.join(tempfile.mkdtemp(), str(uuid.uuid4())) for i in range(0, 4)]
        empiricals = []
        empiricals.append(Empirical(values=values_correct[0:3], log_weights=log_weights_correct[0:3], file_name=file_names[0]))
        empiricals.append(Empirical(values=values_correct[3:5], log_weights=log_weights_correct[3:5], file_name=file_names[1]))
        empiricals.append(Empirical(values=values_correct[5:9], log_weights=log_weights_correct[5:9], file_name=file_names[2]))
        empiricals.append(Empirical(values=values_correct[9:10], log_weights=log_weights_correct[9:10], file_name=file_names[3]))
        [emp.close() for emp in empiricals]
        concat_emp = Empirical(concat_empirical_file_names=file_names)
        concat_emp_mean = float(concat_emp.mean)
        concat_emp_stddev = float(concat_emp.stddev)
        concat_emp_ess = float(concat_emp.effective_sample_size)
        [os.remove(file_name) for file_name in file_names]

        util.eval_print('file_names', 'values_correct', 'log_weights_correct', 'dist_correct_mean', 'concat_emp_mean', 'mean_correct', 'dist_correct_stddev', 'concat_emp_stddev', 'stddev_correct', 'dist_correct_ess', 'concat_emp_ess', 'ess_correct')

        self.assertAlmostEqual(dist_correct_mean, mean_correct, places=1)
        self.assertAlmostEqual(dist_correct_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(dist_correct_ess, ess_correct, places=1)
        self.assertAlmostEqual(concat_emp_mean, mean_correct, places=1)
        self.assertAlmostEqual(concat_emp_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(concat_emp_ess, ess_correct, places=1)

    def test_distributions_empirical_concat_file_to_file(self):
        values_correct = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
        log_weights_correct = [-10, -15, -200, -2, -3, -22, -100, 1, 2, -0.3]
        mean_correct = 7.741360664367676
        stddev_correct = 0.7910336256027222
        ess_correct = 1.9459790014029552

        dist_correct = Empirical(values=values_correct, log_weights=log_weights_correct)
        dist_correct_mean = float(dist_correct.mean)
        dist_correct_stddev = float(dist_correct.stddev)
        dist_correct_ess = float(dist_correct.effective_sample_size)
        file_names = [os.path.join(tempfile.mkdtemp(), str(uuid.uuid4())) for i in range(0, 4)]
        empiricals = []
        empiricals.append(Empirical(values=values_correct[0:3], log_weights=log_weights_correct[0:3], file_name=file_names[0]))
        empiricals.append(Empirical(values=values_correct[3:5], log_weights=log_weights_correct[3:5], file_name=file_names[1]))
        empiricals.append(Empirical(values=values_correct[5:9], log_weights=log_weights_correct[5:9], file_name=file_names[2]))
        empiricals.append(Empirical(values=values_correct[9:10], log_weights=log_weights_correct[9:10], file_name=file_names[3]))
        [emp.close() for emp in empiricals]
        concat_file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        concat_emp = Empirical(concat_empirical_file_names=file_names, file_name=concat_file_name)
        concat_emp.close()
        concat_emp2 = Empirical(file_name=concat_file_name)
        concat_emp_mean = float(concat_emp2.mean)
        concat_emp_stddev = float(concat_emp2.stddev)
        concat_emp_ess = float(concat_emp2.effective_sample_size)
        [os.remove(file_name) for file_name in file_names]
        os.remove(concat_file_name)

        util.eval_print('file_names', 'concat_file_name', 'values_correct', 'log_weights_correct', 'dist_correct_mean', 'concat_emp_mean', 'mean_correct', 'dist_correct_stddev', 'concat_emp_stddev', 'stddev_correct', 'dist_correct_ess', 'concat_emp_ess', 'ess_correct')

        self.assertAlmostEqual(dist_correct_mean, mean_correct, places=1)
        self.assertAlmostEqual(dist_correct_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(dist_correct_ess, ess_correct, places=1)
        self.assertAlmostEqual(concat_emp_mean, mean_correct, places=1)
        self.assertAlmostEqual(concat_emp_stddev, stddev_correct, places=1)
        self.assertAlmostEqual(concat_emp_ess, ess_correct, places=1)

    def test_distributions_empirical_skweness_kurtosis(self):
        values = [Normal(0, 1).sample() for _ in range(empirical_samples)]
        emp = Empirical(values)

        skewness_correct = 0.
        kurtosis_correct = 3.
        skewness = float(emp.skewness)
        kurtosis = float(emp.kurtosis)

        util.eval_print('skewness', 'kurtosis', 'skewness_correct', 'kurtosis_correct')

        self.assertAlmostEqual(skewness_correct, skewness, delta=0.1)
        self.assertAlmostEqual(kurtosis_correct, kurtosis, delta=0.1)

        values = [Beta(0.5, 1).sample() for _ in range(empirical_samples)]
        emp = Empirical(values)

        skewness_correct = 0.638877
        kurtosis_correct = 2.14286
        skewness = float(emp.skewness)
        kurtosis = float(emp.kurtosis)

        util.eval_print('skewness', 'kurtosis', 'skewness_correct', 'kurtosis_correct')

        self.assertAlmostEqual(skewness_correct, skewness, delta=0.1)
        self.assertAlmostEqual(kurtosis_correct, kurtosis, delta=0.1)

    def test_distributions_empirical_median(self):
        values = [Exponential(1.5).sample() for _ in range(empirical_samples)]
        emp = Empirical(values)

        mean_correct = 0.666667
        median_correct = 0.462098
        mean = float(emp.mean)
        median = float(emp.median)

        util.eval_print('mean', 'median', 'mean_correct', 'median_correct')

        self.assertAlmostEqual(mean_correct, mean, delta=0.1)
        self.assertAlmostEqual(median_correct, median, delta=0.1)

    def test_distributions_empirical_trace_index(self):
        model = GaussianWithUnknownMean()
        samples = 1000

        prior_mu_mean_correct = 1

        prior = model.prior(samples)
        prior_mu = prior['mu']
        prior_mu_mean = float(prior_mu.mean)

        util.eval_print('samples', 'prior_mu_mean', 'prior_mu_mean_correct')

        self.assertAlmostEqual(prior_mu_mean_correct, prior_mu_mean, delta=0.5)

    def test_distributions_empirical_reobserve(self):
        file_name_1 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        model = GaussianWithUnknownMean()
        samples = 1500

        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior_reobserved_mean_correct = 4.0111
        posterior_reobserved_stddev_correct = 1.4619

        posterior = model.posterior(samples, observe={'obs0': 8, 'obs1': 9}, file_name=file_name_1)
        posterior_mu = posterior['mu']
        posterior_mean = float(posterior_mu.mean)
        posterior_stddev = float(posterior_mu.stddev)
        posterior_reobserved = posterior.reobserve(likelihood_funcs={'obs0': lambda v, trace: Normal(trace['mu']+1, math.sqrt(5.)), 'obs1': lambda v, trace: Normal(trace['mu']+5, math.sqrt(15.))}, file_name=file_name_2)
        posterior_reobserved_mu = posterior_reobserved['mu']
        posterior_reobserved_mean = float(posterior_reobserved_mu.mean)
        posterior_reobserved_stddev = float(posterior_reobserved_mu.stddev)

        util.eval_print('file_name_1', 'file_name_2', 'samples', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_reobserved_mean', 'posterior_reobserved_mean_correct', 'posterior_reobserved_stddev', 'posterior_reobserved_stddev_correct')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertAlmostEqual(posterior_reobserved_mean, posterior_reobserved_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_reobserved_stddev, posterior_reobserved_stddev_correct, delta=0.75)

    def test_distributions_empirical_reobserve_disk(self):
        model = GaussianWithUnknownMean()
        samples = 1500

        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior_reobserved_mean_correct = 4.0111
        posterior_reobserved_stddev_correct = 1.4619

        posterior = model.posterior(samples, observe={'obs0': 8, 'obs1': 9})
        posterior_mu = posterior['mu']
        posterior_mean = float(posterior_mu.mean)
        posterior_stddev = float(posterior_mu.stddev)
        posterior_reobserved = posterior.reobserve(likelihood_funcs={'obs0': lambda v, trace: Normal(trace['mu']+1, math.sqrt(5.)), 'obs1': lambda v, trace: Normal(trace['mu']+5, math.sqrt(15.))})
        posterior_reobserved_mu = posterior_reobserved['mu']
        posterior_reobserved_mean = float(posterior_reobserved_mu.mean)
        posterior_reobserved_stddev = float(posterior_reobserved_mu.stddev)

        util.eval_print('samples', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_reobserved_mean', 'posterior_reobserved_mean_correct', 'posterior_reobserved_stddev', 'posterior_reobserved_stddev_correct')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertAlmostEqual(posterior_reobserved_mean, posterior_reobserved_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_reobserved_stddev, posterior_reobserved_stddev_correct, delta=0.75)

    def test_distributions_binomial(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_total_count_correct = 10
        dist_probs_correct = 0.2
        dist_logits_correct = -1.3863
        dist_means_correct = 2.
        dist_stddevs_correct = 1.2649
        dist_log_probs_correct = -1.1974

        dist = Binomial(total_count=dist_total_count_correct, probs=dist_probs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_probs = util.to_numpy(dist.probs)
        dist_logits = util.to_numpy(dist.logits)
        dist_total_count = util.to_numpy(dist.total_count)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_probs', 'dist_probs_correct', 'dist_logits', 'dist_logits_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_probs, dist_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_logits, dist_logits_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_total_count, dist_total_count_correct, atol=0.1))

    def test_distributions_weibull(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_concentration_correct = 0.5
        dist_scale_correct = 1.1
        dist_means_correct = 2.2
        dist_stddevs_correct = 4.9193
        dist_log_probs_correct = -2.5492

        dist = Weibull(scale=dist_scale_correct, concentration=dist_concentration_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_concentration = util.to_numpy(dist.concentration)
        dist_scale = util.to_numpy(dist.scale)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_concentration', 'dist_concentration_correct', 'dist_scale', 'dist_scale_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.33))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_concentration, dist_concentration_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_scale, dist_scale_correct, atol=0.1))

    def test_distributions_von_mises(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_locs_correct = 3.1415
        dist_concentration_correct = 2.
        dist_means_correct = 3.1415
        dist_stddevs_correct = 0.5498
        dist_log_probs_correct = -0.6619

        dist = VonMises(loc=dist_locs_correct, concentration=dist_concentration_correct)
        dist_concentration = util.to_numpy(dist.concentration)
        dist_locs = util.to_numpy(dist.loc)
        dist_means = util.to_numpy(dist.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_concentration', 'dist_concentration_correct', 'dist_locs', 'dist_locs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_concentration, dist_concentration_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_locs, dist_locs_correct, atol=0.1))

    def test_distributions_gamma(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_concentrations_correct = 0.5
        dist_rates_correct = 1.2
        dist_means_correct = 0.4167
        dist_stddevs_correct = 0.5893
        dist_log_probs_correct = -0.5435

        dist = Gamma(dist_concentrations_correct, dist_rates_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_concentrations = util.to_numpy(dist.concentration)
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_concentrations', 'dist_concentrations_correct', 'dist_rates', 'dist_rates_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_concentrations, dist_concentrations_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))

    def test_distributions_gamma_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_concentrations_correct = [[0.5], [2.7]]
        dist_rates_correct = [[1.2], [5.5]]
        dist_means_correct = [[0.4167], [0.4909]]
        dist_stddevs_correct = [[0.5893], [0.2988]]
        dist_log_probs_correct = [[-0.5435], [0.2585]]

        dist = Gamma(dist_concentrations_correct, dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()
        dist_concentrations = util.to_numpy(dist.concentration)
        dist_rates = util.to_numpy(dist.rate)

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_concentrations', 'dist_concentrations_correct', 'dist_rates', 'dist_rates_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_concentrations, dist_concentrations_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))

    def test_distributions_log_normal(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_loc_correct = 0.5
        dist_scale_correct = 0.2
        dist_means_correct = 1.6820
        dist_stddevs_correct = 0.3398
        dist_log_probs_correct = 0.1655

        dist = LogNormal(dist_loc_correct, dist_scale_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()
        dist_loc = dist.loc
        dist_scale = dist.scale

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_loc', 'dist_loc_correct', 'dist_scale', 'dist_scale_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_loc, dist_loc_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_scale, dist_scale_correct, atol=0.1))

    def test_distributions_log_normal_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_loc_correct = [[0.5], [1.4]]
        dist_scale_correct = [[0.2], [0.5]]
        dist_means_correct = [[1.6820], [4.5951]]
        dist_stddevs_correct = [[0.3398], [2.4489]]
        dist_log_probs_correct = [[0.1655], [-1.7820]]

        dist = LogNormal(dist_loc_correct, dist_scale_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()
        dist_loc = dist.loc
        dist_scale = dist.scale

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct', 'dist_loc', 'dist_loc_correct', 'dist_scale', 'dist_scale_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_loc, dist_loc_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_scale, dist_scale_correct, atol=0.1))

    def test_distributions_normal(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_correct = 0
        dist_stddevs_correct = 1
        dist_log_probs_correct = -0.918939

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_normal_batched_2(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_prob_shape_correct = torch.Size([2])
        dist_means_correct = [0, 2]
        dist_stddevs_correct = [1, 3]
        dist_log_probs_correct = [-0.918939, -2.01755]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_normal_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_means_correct = [[0], [2]]
        dist_stddevs_correct = [[1], [3]]
        dist_log_probs_correct = [[-0.918939], [-2.01755]]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_normal_batched_2_3(self):
        dist_batch_shape_correct = torch.Size([2, 3])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 3])
        dist_log_prob_shape_correct = torch.Size([2, 3])
        dist_means_correct = [[0, 2, 0], [2, 0, 2]]
        dist_stddevs_correct = [[1, 3, 1], [3, 1, 3]]
        dist_log_probs_correct = [[-0.918939, -2.01755, -0.918939], [-2.01755, -0.918939, -2.01755]]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_truncated_normal(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_non_truncated_correct = 2
        dist_stddevs_non_truncated_correct = 3
        dist_means_correct = 0.901189
        dist_stddevs_correct = 1.95118
        dist_lows_correct = -4
        dist_highs_correct = 4
        dist_log_probs_correct = -1.69563

        dist = TruncatedNormal(dist_means_non_truncated_correct, dist_stddevs_non_truncated_correct, dist_lows_correct, dist_highs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_non_truncated_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_non_truncated_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_truncated_normal_batched_2(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_prob_shape_correct = torch.Size([2])
        dist_means_non_truncated_correct = [0, 2]
        dist_stddevs_non_truncated_correct = [1, 3]
        dist_means_correct = [0, 0.901189]
        dist_stddevs_correct = [0.53956, 1.95118]
        dist_lows_correct = [-1, -4]
        dist_highs_correct = [1, 4]
        dist_log_probs_correct = [-0.537223, -1.69563]

        dist = TruncatedNormal(dist_means_non_truncated_correct, dist_stddevs_non_truncated_correct, dist_lows_correct, dist_highs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_non_truncated_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_non_truncated_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_truncated_normal_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_means_non_truncated_correct = [[0], [2]]
        dist_stddevs_non_truncated_correct = [[1], [3]]
        dist_means_correct = [[0], [0.901189]]
        dist_stddevs_correct = [[0.53956], [1.95118]]
        dist_lows_correct = [[-1], [-4]]
        dist_highs_correct = [[1], [4]]
        dist_log_probs_correct = [[-0.537223], [-1.69563]]

        dist = TruncatedNormal(dist_means_non_truncated_correct, dist_stddevs_non_truncated_correct, dist_lows_correct, dist_highs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_non_truncated_correct))
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_prob_shape = dist.log_prob(dist_means_non_truncated_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_truncated_normal_clamped_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_means_non_truncated = [[0], [2]]
        dist_means_non_truncated_correct = [[0.5], [1]]
        dist_stddevs_non_truncated = [[1], [3]]
        dist_means_correct = [[0.744836], [-0.986679]]
        dist_stddevs_correct = [[0.143681], [1.32416]]
        dist_lows_correct = [[0.5], [-4]]
        dist_highs_correct = [[1], [1]]
        dist_log_prob_arguments = [[0.75], [-3]]
        dist_log_probs_correct = [[0.702875], [-2.11283]]

        dist = TruncatedNormal(dist_means_non_truncated, dist_stddevs_non_truncated, dist_lows_correct, dist_highs_correct, clamp_mean_between_low_high=True)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means_non_truncated = util.to_numpy(dist._mean_non_truncated)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_probs = util.to_numpy(dist.log_prob(dist_log_prob_arguments))
        dist_log_prob_shape = dist.log_prob(dist_log_prob_arguments).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_means_non_truncated', 'dist_means_non_truncated_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means_non_truncated, dist_means_non_truncated_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_categorical(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_correct = 1.6
        dist_stddevs_correct = 0.666
        dist_log_probs_correct = -2.30259

        dist = Categorical([0.1, 0.2, 0.7])
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample().float() for i in range(empirical_samples)])
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(0))
        dist_log_prob_shape = dist.log_prob(0).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_categorical_batched_2(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_prob_shape_correct = torch.Size([2])
        dist_means_correct = [1.6, 1.1]
        dist_stddevs_correct = [0.666, 0.7]
        dist_log_probs_correct = [-2.30259, -0.693147]

        dist = Categorical([[0.1, 0.2, 0.7],
                            [0.2, 0.5, 0.3]])

        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample().float() for i in range(empirical_samples)])
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob([0, 1]))
        dist_log_prob_shape = dist.log_prob([0, 1]).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_categorical_logits(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_correct = 1.6
        dist_stddevs_correct = 0.666
        dist_log_probs_correct = -2.30259

        dist = Categorical(logits=[-2.30259, -1.60944, -0.356675])
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample().float() for i in range(empirical_samples)])
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(0))
        dist_log_prob_shape = dist.log_prob(0).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_uniform(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_correct = 0.5
        dist_stddevs_correct = 0.288675
        dist_lows_correct = 0
        dist_highs_correct = 1
        dist_log_probs_correct = 0

        dist = Uniform(dist_lows_correct, dist_highs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_lows = util.to_numpy(dist.low)
        dist_highs = util.to_numpy(dist.high)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_lows', 'dist_lows_correct', 'dist_highs', 'dist_highs_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_uniform_batched_4_1(self):
        dist_batch_shape_correct = torch.Size([4, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([4, 1])
        dist_log_prob_shape_correct = torch.Size([4, 1])
        dist_means_correct = [[0.5], [7.5], [0.5], [0.5]]
        dist_stddevs_correct = [[0.288675], [1.44338], [0.288675], [0.288675]]
        dist_lows_correct = [[0], [5], [0], [0]]
        dist_highs_correct = [[1], [10], [1], [1]]
        dist_values = [[0.5], [7.5], [0], [1]]
        dist_log_probs_correct = [[0], [-1.60944], [0.], [float('-inf')]]

        dist = Uniform(dist_lows_correct, dist_highs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_lows = util.to_numpy(dist.low)
        dist_highs = util.to_numpy(dist.high)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_values))
        dist_log_prob_shape = dist.log_prob(dist_values).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_lows', 'dist_lows_correct', 'dist_highs', 'dist_highs_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_poisson(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_means_correct = 4
        dist_stddevs_correct = math.sqrt(4)
        dist_rates_correct = 4
        dist_log_probs_correct = -1.63288

        dist = Poisson(dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_poisson_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_means_correct = [[4], [100]]
        dist_stddevs_correct = [[math.sqrt(4)], [math.sqrt(100)]]
        dist_rates_correct = [[4], [100]]
        dist_log_probs_correct = [[-1.63288], [-3.22236]]

        dist = Poisson(dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_poisson_batched_1_3(self):
        dist_batch_shape_correct = torch.Size([1, 3])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([1, 3])
        dist_log_prob_shape_correct = torch.Size([1, 3])
        dist_means_correct = [[1, 2, 15]]
        dist_stddevs_correct = [[math.sqrt(1), math.sqrt(2), math.sqrt(15)]]
        dist_rates_correct = [[1, 2, 15]]
        dist_log_probs_correct = [[-1, -1.30685, -2.27852]]

        dist = Poisson(dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_exponential(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_rates_correct = 4
        dist_means_correct = 0.25
        dist_stddevs_correct = 0.25
        dist_log_probs_correct = 0.3863

        dist = Exponential(dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_exponential_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_rates_correct = [[1.2], [3.]]
        dist_means_correct = [[0.8333], [0.3333]]
        dist_stddevs_correct = [[0.8333], [0.3333]]
        dist_log_probs_correct = [[-0.8177], [0.0986]]

        dist = Exponential(dist_rates_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_beta(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_concentration1s_correct = 2
        dist_concentration0s_correct = 5
        dist_means_correct = 0.285714
        dist_stddevs_correct = 0.159719
        dist_log_probs_correct = 0.802545

        dist = Beta(dist_concentration1s_correct, dist_concentration0s_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_beta_batched_4_1(self):
        dist_batch_shape_correct = torch.Size([4, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([4, 1])
        dist_log_prob_shape_correct = torch.Size([4, 1])
        dist_concentration1s_correct = [[0.5], [7.5], [7.5], [7.5]]
        dist_concentration0s_correct = [[0.75], [2.5], [2.5], [2.5]]
        dist_means_correct = [[0.4], [0.75], [0.75], [0.75]]
        dist_stddevs_correct = [[0.326599], [0.130558], [0.130558], [0.130558]]
        dist_values = [[0.415584], [0.807999], [0.], [1.]]
        dist_log_probs_correct = [[-0.300597], [1.12163], [float('-inf')], [float('-inf')]]

        dist = Beta(dist_concentration1s_correct, dist_concentration0s_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_values))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_beta_low_high(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_concentration1s_correct = 2
        dist_concentration0s_correct = 3
        dist_lows_correct = -2
        dist_highs_correct = 5
        dist_means_correct = 0.8
        dist_stddevs_correct = 1.4
        dist_log_probs_correct = 0.546965

        dist = Beta(dist_concentration1s_correct, dist_concentration0s_correct, low=dist_lows_correct, high=dist_highs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_lows = util.to_numpy(dist.low)
        dist_lows_empirical = util.to_numpy(dist_empirical.min)
        dist_highs = util.to_numpy(dist.high)
        dist_highs_empirical = util.to_numpy(dist_empirical.max)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_lows', 'dist_lows_empirical', 'dist_lows_correct', 'dist_highs', 'dist_highs_empirical', 'dist_highs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows_empirical, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs_empirical, dist_highs_correct, atol=0.33))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_beta_low_high_batched_2(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_prob_shape_correct = torch.Size([2])
        dist_concentration1s_correct = [2, 2]
        dist_concentration0s_correct = [3, 3]
        dist_lows_correct = [-2, 3]
        dist_highs_correct = [5, 4]
        dist_means_correct = [0.8, 3.4]
        dist_stddevs_correct = [1.4, 0.2]
        dist_log_probs_correct = [0.546965, 0.546965]

        dist = Beta(dist_concentration1s_correct, dist_concentration0s_correct, low=dist_lows_correct, high=dist_highs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_lows = util.to_numpy(dist.low)
        dist_lows_empirical = util.to_numpy([dist_empirical.map(lambda x: x[0]).min, dist_empirical.map(lambda x: x[1]).min])
        dist_highs = util.to_numpy(dist.high)
        dist_highs_empirical = util.to_numpy([dist_empirical.map(lambda x: x[0]).max, dist_empirical.map(lambda x: x[1]).max])
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_lows', 'dist_lows_empirical', 'dist_lows_correct', 'dist_highs', 'dist_highs_empirical', 'dist_highs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows_empirical, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs_empirical, dist_highs_correct, atol=0.33))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_beta_low_high_batched_2_1(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
        dist_log_prob_shape_correct = torch.Size([2, 1])
        dist_concentration1s_correct = [[2], [2]]
        dist_concentration0s_correct = [[3], [3]]
        dist_lows_correct = [[-2], [3]]
        dist_highs_correct = [[5], [4]]
        dist_means_correct = [[0.8], [3.4]]
        dist_stddevs_correct = [[1.4], [0.2]]
        dist_log_probs_correct = [[0.546965], [0.546965]]

        dist = Beta(dist_concentration1s_correct, dist_concentration0s_correct, low=dist_lows_correct, high=dist_highs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_lows = util.to_numpy(dist.low)
        dist_lows_empirical = util.to_numpy([[dist_empirical.map(lambda x: x[0]).min], [dist_empirical.map(lambda x: x[1]).min]])
        dist_highs = util.to_numpy(dist.high)
        dist_highs_empirical = util.to_numpy([[dist_empirical.map(lambda x: x[0]).max], [dist_empirical.map(lambda x: x[1]).max]])
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_lows', 'dist_lows_empirical', 'dist_lows_correct', 'dist_highs', 'dist_highs_empirical', 'dist_highs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows_empirical, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs_empirical, dist_highs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_bernoulli(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_probs_correct = 0.2
        dist_means_correct = 0.2
        dist_stddevs_correct = 0.4
        dist_log_probs_correct = -0.5004

        dist = Bernoulli(probs=dist_probs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_bernoulli_batched_4_1(self):
        dist_batch_shape_correct = torch.Size([4, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([4, 1])
        dist_log_prob_shape_correct = torch.Size([4, 1])
        dist_probs_correct = [[0.5], [0.2], [0.95], [0.1]]
        dist_means_correct = [[0.5], [0.2], [0.95], [0.1]]
        dist_stddevs_correct = [[0.5], [0.4], [0.2179], [0.3]]
        dist_values = [[0], [0], [0.], [1]]
        dist_log_probs_correct = [[-0.6931], [-0.2231], [-2.9957], [-2.3026]]

        dist = Bernoulli(probs=dist_probs_correct)
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_values))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_mixture(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_prob_shape_correct = torch.Size()
        dist_1 = Normal(0, 0.1)
        dist_2 = Normal(2, 0.1)
        dist_3 = Normal(3, 0.1)
        dist_means_correct = 0.7
        dist_stddevs_correct = 1.10454
        dist_log_probs_correct = -23.473

        dist = Mixture([dist_1, dist_2, dist_3], probs=[0.7, 0.2, 0.1])
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_mixture_batched_2(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_prob_shape_correct = torch.Size([2])
        dist_1 = Normal([0, 1], [0.1, 1])
        dist_2 = Normal([2, 5], [0.1, 1])
        dist_3 = Normal([3, 10], [0.1, 1])
        dist_means_correct = [0.7, 8.1]
        dist_stddevs_correct = [1.10454, 3.23883]
        dist_log_probs_correct = [-23.473, -3.06649]

        dist = Mixture([dist_1, dist_2, dist_3], probs=[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))
        dist_log_prob_shape = dist.log_prob(dist_means_correct).size()

        util.eval_print('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_prob_shape', 'dist_log_prob_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertEqual(dist_log_prob_shape, dist_log_prob_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_distributions_repr(self):
        def exec_return(code):
            exec('global exec_return_result; exec_return_result = %s' % code)
            global exec_return_result
            return exec_return_result

        dists = []
        dists.append(Bernoulli(0.5))
        dists.append(Binomial(2, 0.5))
        dists.append(Categorical([0.2, 0.3, 0.5]))
        dists.append(Exponential(1.5))
        dists.append(Gamma(0, 1))
        dists.append(LogNormal(0, 1))
        dists.append(Mixture([Normal(0, 1), Normal(2, 3)], [0.4, 0.6]))
        dists.append(Normal(0, 1))
        dists.append(Poisson(5))
        dists.append(TruncatedNormal(0, 1, 0.5, 0.6))
        dists.append(Uniform(0.2, 0.6))
        dists.append(VonMises(0.1, 1.1))
        dists.append(Weibull(0.1, 1.1))

        dists_repr = list(map(repr, dists))
        dists_repr_exec = list(map(exec_return, dists_repr))
        dists_repr_exec_repr = list(map(repr, dists_repr_exec))

        util.eval_print('dists', 'dists_repr', 'dists_repr_exec')

        self.assertEqual(dists_repr, dists_repr_exec_repr)

    def test_distribution_factor(self):
        log_prob_correct = 0.56
        dist = Factor(log_prob_correct)
        log_prob = float(dist.log_prob())
        util.eval_print('dist', 'log_prob', 'log_prob_correct')
        self.assertTrue(np.allclose(log_prob_correct, log_prob))

    def test_distribution_factor_func(self):
        log_prob_func = lambda x: x*x
        dist = Factor(log_prob_func=log_prob_func)
        value = 0.66
        log_prob = float(dist.log_prob(value))
        log_prob_correct = float(log_prob_func(value))
        util.eval_print('dist', 'log_prob', 'log_prob_correct')
        self.assertTrue(np.allclose(log_prob_correct, log_prob))

if __name__ == '__main__':
    pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
