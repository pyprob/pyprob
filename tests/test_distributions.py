import unittest
import torch
import numpy as np
import os
import math
import uuid
import tempfile

from pyprob import util
from pyprob.distributions import Distribution, Empirical, Normal, Categorical


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
        # dist_sample_shape_correct = []

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

        util.debug('dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct', 'dist_min', 'dist_min_correct', 'dist_max', 'dist_max_correct')

        # self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_expectation_sin, dist_expectation_sin_correct, places=1)
        self.assertAlmostEqual(dist_map_sin_mean, dist_map_sin_mean_correct, places=1)
        self.assertAlmostEqual(dist_min, dist_min_correct, places=1)
        self.assertAlmostEqual(dist_max, dist_max_correct, places=1)

    def test_dist_empirical_resample(self):
        dist_means_correct = [2]
        dist_stddevs_correct = [5]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_empirical = dist_empirical.resample(int(empirical_samples/2))
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        util.debug('dist_means_empirical', 'dist_means_correct', 'dist_stddevs_empirical', 'dist_stddevs_correct')

        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))

    def test_dist_empirical_combine_uniform_weights(self):
        dist1_mean_correct = 1
        dist1_stddev_correct = 3
        dist2_mean_correct = 5
        dist2_stddev_correct = 2
        dist3_mean_correct = -2.5
        dist3_stddev_correct = 1.2
        dist_combined_mean_correct = 1.16667
        dist_combined_stddev_correct = 3.76858

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
        dist_combined_empirical = Empirical.combine([dist1_empirical, dist2_empirical, dist3_empirical])
        dist_combined_mean_empirical = float(dist_combined_empirical.mean)
        dist_combined_stddev_empirical = float(dist_combined_empirical.stddev)

        util.debug('dist1_mean_empirical', 'dist1_stddev_empirical', 'dist1_mean_correct', 'dist1_stddev_correct', 'dist2_mean_empirical', 'dist2_stddev_empirical', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist3_mean_empirical', 'dist3_stddev_empirical', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist_combined_mean_empirical', 'dist_combined_stddev_empirical', 'dist_combined_mean_correct', 'dist_combined_stddev_correct')

        self.assertAlmostEqual(dist1_mean_empirical, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_stddev_empirical, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_mean_empirical, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_stddev_empirical, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_mean_empirical, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_stddev_empirical, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_mean_empirical, dist_combined_mean_correct, places=1)
        self.assertAlmostEqual(dist_combined_stddev_empirical, dist_combined_stddev_correct, places=1)

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

        util.debug('file_name', 'dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_expectation_sin', 'dist_expectation_sin_correct', 'dist_map_sin_mean', 'dist_map_sin_mean_correct')

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
        dist_means_correct = 0
        dist_stddevs_correct = 1
        dist_log_probs_correct = -0.918939

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

        util.debug('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_batched(self):
        dist_batch_shape_correct = torch.Size([2, 1])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 1])
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

        util.debug('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_batched_2(self):
        dist_batch_shape_correct = torch.Size([2, 3])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2, 3])
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

        util.debug('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_categorical(self):
        dist_batch_shape_correct = torch.Size()
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size()
        dist_log_probs_correct = -2.30259

        dist = Categorical([0.1, 0.2, 0.7])

        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_probs = util.to_numpy(dist.log_prob(0))

        util.debug('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_categorical_batched(self):
        dist_batch_shape_correct = torch.Size([2])
        dist_event_shape_correct = torch.Size()
        dist_sample_shape_correct = torch.Size([2])
        dist_log_probs_correct = [-2.30259, -0.693147]

        dist = Categorical([[0.1, 0.2, 0.7],
                            [0.2, 0.5, 0.3]])

        dist_batch_shape = dist.batch_shape
        dist_event_shape = dist.event_shape
        dist_sample_shape = dist.sample().size()
        dist_log_probs = util.to_numpy(dist.log_prob([0, 1]))

        util.debug('dist_batch_shape', 'dist_batch_shape_correct', 'dist_event_shape', 'dist_event_shape_correct', 'dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_batch_shape, dist_batch_shape_correct)
        self.assertEqual(dist_event_shape, dist_event_shape_correct)
        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))
