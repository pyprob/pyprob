import unittest
import torch
import numpy as np
import os
import math
import uuid
import tempfile

import pyprob
from pyprob import util
from pyprob.distributions import Distribution, Empirical, Normal, Categorical, Uniform, Poisson, Beta, Mixture, TruncatedNormal


empirical_samples = 20000


class DistributionsTestCase(unittest.TestCase):
    def test_dist_empirical(self):
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

    def test_dist_empirical_copy(self):
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

    def test_dist_empirical_disk(self):
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

    def test_dist_empirical_disk_append(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        dist_means_correct = -1.2
        dist_stddevs_correct = 1.4
        dist_empirical_length_correct = 1000

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical(file_name=file_name)
        dist_empirical.add_sequence([dist.sample() for i in range(500)])
        dist_empirical.finalize()
        dist_empirical.close()
        dist_empirical_2 = Empirical(file_name=file_name)
        dist_empirical_2.add_sequence([dist.sample() for i in range(500)])
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

    def test_dist_empirical_combine_duplicates(self):
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

    def test_dist_empirical_numpy(self):
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

    def test_dist_empirical_resample(self):
        dist_means_correct = [2]
        dist_stddevs_correct = [5]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_empirical = dist_empirical.resample(int(empirical_samples/2))
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        util.eval_print('dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct')

        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))

    def test_dist_empirical_slice_and_index(self):
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

    def test_dist_empirical_combine_uniform_weights(self):
        dist1_mean_correct = 1
        dist1_stddev_correct = 3
        dist2_mean_correct = 5
        dist2_stddev_correct = 2
        dist3_mean_correct = -2.5
        dist3_stddev_correct = 1.2
        dist_combined_mean_correct = 1.16667
        dist_combined_stddev_correct = 3.76858

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

        dist_combined_empirical = Empirical.combine(empirical_distributions=[dist1_empirical, dist2_empirical, dist3_empirical])
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)

        util.eval_print('dist1_mean_empirical', 'dist1_stddev_empirical', 'dist1_mean_correct', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_stddev_empirical', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_stddev_empirical', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_stddev_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=1)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=1)

    def test_dist_empirical_disk_combine_uniform_weights(self):
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

        dist1 = Normal(dist1_mean_correct, dist1_stddev_correct)
        dist1_empirical = Empirical([dist1.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_1)
        dist1_mean_empirical = float(dist1_empirical.mean)
        dist1_stddev_empirical = float(dist1_empirical.stddev)

        dist2 = Normal(dist2_mean_correct, dist2_stddev_correct)
        dist2_empirical = Empirical([dist2.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_2)
        dist2_mean_empirical = float(dist2_empirical.mean)
        dist2_stddev_empirical = float(dist2_empirical.stddev)

        dist3 = Normal(dist3_mean_correct, dist3_stddev_correct)
        dist3_empirical = Empirical([dist3.sample() for i in range(int(empirical_samples / 10))], file_name=file_name_3)
        dist3_mean_empirical = float(dist3_empirical.mean)
        dist3_stddev_empirical = float(dist3_empirical.stddev)

        dist_combined_empirical = Empirical.combine(empirical_distributions=[dist1_empirical, dist2_empirical, dist3_empirical], file_name=file_name_combined)
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)

        util.eval_print('dist1_mean_empirical', 'dist1_stddev_empirical', 'dist1_mean_correct', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_stddev_empirical', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_stddev_empirical', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_stddev_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=0)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=0)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=0)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=0)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=0)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=0)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=0)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=0)

    def test_dist_empirical_save_load(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        values = util.to_tensor([1, 2, 3])
        log_weights = util.to_tensor([1, 2, 3])
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675
        dist_expectation_sin_correct = 0.3921678960323334
        dist_map_sin_mean_correct = 0.3921678960323334

        dist_on_file = Empirical(values, log_weights)
        dist_on_file.save(file_name)
        dist = Distribution.load(file_name)
        os.remove(file_name)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_expectation_sin = float(dist.expectation(torch.sin))
        dist_map_sin_mean = float(dist.map(torch.sin).mean)

        util.eval_print('file_name', 'dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_expectation_sin, dist_expectation_sin_correct, places=1)
        self.assertAlmostEqual(dist_map_sin_mean, dist_map_sin_mean_correct, places=1)

    def test_dist_normal(self):
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

    def test_dist_normal_batched_2(self):
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

    def test_dist_normal_batched_2_1(self):
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

    def test_dist_normal_batched_2_3(self):
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

    def test_dist_truncated_normal(self):
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

    def test_dist_truncated_normal_batched_2(self):
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

    def test_dist_truncated_normal_batched_2_1(self):
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

    def test_dist_truncated_normal_clamped_batched_2_1(self):
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

    def test_dist_categorical(self):
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

    def test_dist_categorical_batched_2(self):
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

    def test_dist_categorical_logits(self):
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

    def test_dist_uniform(self):
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

    def test_dist_uniform_batched_4_1(self):
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

    def test_dist_poisson(self):
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

    def test_dist_poisson_batched_2_1(self):
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
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson_batched_1_3(self):
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

    def test_dist_beta(self):
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

    def test_dist_beta_batched_4_1(self):
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

    def test_dist_beta_low_high(self):
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
        self.assertTrue(np.allclose(dist_highs_empirical, dist_highs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_beta_low_high_batched_2(self):
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
        self.assertTrue(np.allclose(dist_highs_empirical, dist_highs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_beta_low_high_batched_2_1(self):
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

    def test_dist_mixture(self):
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

    def test_dist_mixture_batched_2(self):
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


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
