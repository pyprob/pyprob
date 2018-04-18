import unittest
import torch
from torch.autograd import Variable
import numpy as np
import uuid
import tempfile
import os
import math

import pyprob
from pyprob import util
from pyprob.distributions import Distribution, Categorical, Empirical, Mixture, Normal, TruncatedNormal, Uniform, Poisson


empirical_samples = 10000


class DistributionsTestCase(unittest.TestCase):
    def test_dist_empirical(self):
        values = Variable(util.Tensor([1, 2, 3]))
        log_weights = Variable(util.Tensor([1, 2, 3]))
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675
        dist_expectation_sin_correct = 0.3921678960323334
        dist_map_sin_mean_correct = 0.3921678960323334
        dist_min_correct = 1
        dist_max_correct = 3

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

    def test_dist_empirical_combine(self):
        dist1_mean_correct = 1
        dist1_stddev_correct = 3
        dist2_mean_correct = 5
        dist2_stddev_correct = 2
        dist3_mean_correct = -2.5
        dist3_stddev_correct = 1.2

        dist1 = Normal(dist1_mean_correct, dist1_stddev_correct)
        dist1_empirical = Empirical([dist1.sample() for i in range(empirical_samples)])
        dist1_empirical_mean = float(dist1_empirical.mean)
        dist1_empirical_stddev = float(dist1_empirical.stddev)
        dist2 = Normal(dist2_mean_correct, dist2_stddev_correct)
        dist2_empirical = Empirical([dist2.sample() for i in range(empirical_samples)])
        dist2_empirical_mean = float(dist2_empirical.mean)
        dist2_empirical_stddev = float(dist2_empirical.stddev)
        dist3 = Normal(dist3_mean_correct, dist3_stddev_correct)
        dist3_empirical = Empirical([dist3.sample() for i in range(empirical_samples)])
        dist3_empirical_mean = float(dist3_empirical.mean)
        dist3_empirical_stddev = float(dist3_empirical.stddev)
        dist_combined = Mixture([dist1, dist2, dist3])
        dist_combined_mean = float(dist_combined.mean)
        dist_combined_stddev = float(dist_combined.stddev)
        dist_combined_empirical = dist1_empirical.combine(dist2_empirical).combine(dist3_empirical)
        dist_combined_empirical_mean = float(dist_combined_empirical.mean)
        dist_combined_empirical_stddev = float(dist_combined_empirical.stddev)

        util.debug('dist1_mean_correct', 'dist1_stddev_correct', 'dist1_empirical_mean', 'dist1_empirical_stddev', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist2_empirical_mean', 'dist2_empirical_stddev', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist3_empirical_mean', 'dist3_empirical_stddev', 'dist_combined_mean', 'dist_combined_stddev', 'dist_combined_empirical_mean', 'dist_combined_empirical_stddev')

        self.assertAlmostEqual(dist1_empirical_mean, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_empirical_stddev, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_empirical_mean, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_empirical_stddev, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_empirical_mean, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_empirical_stddev, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_empirical_mean, dist_combined_mean, places=1)
        self.assertAlmostEqual(dist_combined_empirical_stddev, dist_combined_stddev, places=1)

    def test_dist_categorical(self):
        dist_sample_shape_correct = [1]
        dist_log_probs_correct = [-2.30259]

        dist = Categorical([0.1, 0.2, 0.7])

        dist_sample_shape = list(dist.sample().size())
        dist_log_probs = util.to_numpy(dist.log_prob(0))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_categorical_batched(self):
        dist_sample_shape_correct = [2]
        dist_log_probs_correct = [[-2.30259], [-0.693147]]

        dist = Categorical([[0.1, 0.2, 0.7],
                            [0.2, 0.5, 0.3]])

        dist_sample_shape = list(dist.sample().size())
        dist_log_probs = util.to_numpy(dist.log_prob([[0, 1]]))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_mixture(self):
        dist_sample_shape_correct = [1]
        dist_1 = Normal(0, 0.1)
        dist_2 = Normal(2, 0.1)
        dist_3 = Normal(3, 0.1)
        dist_means_correct = [0.7]
        dist_stddevs_correct = [1.10454]
        dist_log_probs_correct = [-23.473]

        dist = Mixture([dist_1, dist_2, dist_3], probs=[0.7, 0.2, 0.1])

        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        # print(dist.log_prob([2,2]))
        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_mixture_batched(self):
        dist_sample_shape_correct = [2, 1]
        dist_1 = Normal([[0], [1]], [[0.1], [1]])
        dist_2 = Normal([[2], [5]], [[0.1], [1]])
        dist_3 = Normal([[3], [10]], [[0.1], [1]])
        dist_means_correct = [[0.7], [8.1]]
        dist_stddevs_correct = [[1.10454], [3.23883]]
        dist_log_probs_correct = [[-23.473], [-3.06649]]

        dist = Mixture([dist_1, dist_2, dist_3], probs=[[0.7, 0.2, 0.1],[0.1, 0.2, 0.7]])

        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal(self):
        dist_sample_shape_correct = [1]
        dist_means_correct = [0]
        dist_stddevs_correct = [1]
        dist_log_probs_correct = [-0.918939]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_batched(self):
        dist_sample_shape_correct = [2, 1]
        dist_means_correct = [[0], [2]]
        dist_stddevs_correct = [[1], [3]]
        dist_log_probs_correct = [[-0.918939], [-2.01755]]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_multivariate(self):
        dist_sample_shape_correct = [1, 3]
        dist_means_correct = [[0, 2, 0]]
        dist_stddevs_correct = [[1, 3, 1]]
        dist_log_probs_correct = [sum([-0.918939, -2.01755, -0.918939])]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_multivariate_from_flat_params(self):
        dist_sample_shape_correct = [1, 3]
        dist_means_correct = [[0, 2, 0]]
        dist_stddevs_correct = [[1, 3, 1]]
        dist_log_probs_correct = [sum([-0.918939, -2.01755, -0.918939])]

        dist = Normal(dist_means_correct[0], dist_stddevs_correct[0])
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct[0]))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_normal_multivariate_batched(self):
        dist_sample_shape_correct = [2, 3]
        dist_means_correct = [[0, 2, 0], [2, 0, 2]]
        dist_stddevs_correct = [[1, 3, 1], [3, 1, 3]]
        dist_log_probs_correct = [[sum([-0.918939, -2.01755, -0.918939])], [sum([-2.01755, -0.918939, -2.01755])]]

        dist = Normal(dist_means_correct, dist_stddevs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_truncated_normal(self):
        dist_sample_shape_correct = [1]
        dist_means_non_truncated_correct = [2]
        dist_stddevs_non_truncated_correct = [3]
        dist_means_correct = [0.901189]
        dist_stddevs_correct = [1.95118]
        dist_lows_correct = [-4]
        dist_highs_correct = [4]
        dist_log_probs_correct = [-1.69563]

        dist = TruncatedNormal(dist_means_non_truncated_correct, dist_stddevs_non_truncated_correct, dist_lows_correct, dist_highs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_non_truncated_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_truncated_normal_batched(self):
        dist_sample_shape_correct = [2, 1]
        dist_means_non_truncated_correct = [[0], [2]]
        dist_stddevs_non_truncated_correct = [[1], [3]]
        dist_means_correct = [[0], [0.901189]]
        dist_stddevs_correct = [[0.53956], [1.95118]]
        dist_lows_correct = [[-1], [-4]]
        dist_highs_correct = [[1], [4]]
        dist_log_probs_correct = [[-0.537223], [-1.69563]]

        dist = TruncatedNormal(dist_means_non_truncated_correct, dist_stddevs_non_truncated_correct, dist_lows_correct, dist_highs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_non_truncated_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_truncated_normal_clamped_batched(self):
        dist_sample_shape_correct = [2, 1]
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
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means_non_truncated = util.to_numpy(dist._mean_non_truncated)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_log_prob_arguments))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means_non_truncated', 'dist_means_non_truncated_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means_non_truncated, dist_means_non_truncated_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_uniform(self):
        dist_sample_shape_correct = [1]
        dist_means_correct = [0.5]
        dist_stddevs_correct = [0.288675]
        dist_lows_correct = [0]
        dist_highs_correct = [1]
        dist_log_probs_correct = [0]

        dist = Uniform(dist_lows_correct, dist_highs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_lows = util.to_numpy(dist.low)
        dist_highs = util.to_numpy(dist.high)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_lows', 'dist_lows_correct', 'dist_highs', 'dist_highs_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_uniform_batched(self):
        dist_sample_shape_correct = [2, 1]
        dist_means_correct = [[0.5], [7.5]]
        dist_stddevs_correct = [[0.288675], [1.44338]]
        dist_lows_correct = [[0], [5]]
        dist_highs_correct = [[1], [10]]
        dist_log_probs_correct = [[0], [-1.60944]]

        dist = Uniform(dist_lows_correct, dist_highs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_lows = util.to_numpy(dist.low)
        dist_highs = util.to_numpy(dist.high)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_lows', 'dist_lows_correct', 'dist_highs', 'dist_highs_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_lows, dist_lows_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_highs, dist_highs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson(self):
        dist_sample_shape_correct = [1]
        dist_means_correct = [4]
        dist_stddevs_correct = [math.sqrt(4)]
        dist_rates_correct = [4]
        dist_log_probs_correct = [-1.63288]

        dist = Poisson(dist_rates_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson_batched(self):
        dist_sample_shape_correct = [2, 1]
        dist_means_correct = [[4], [100]]
        dist_stddevs_correct = [[math.sqrt(4)], [math.sqrt(100)]]
        dist_rates_correct = [[4], [100]]
        dist_log_probs_correct = [[-1.63288], [-3.22236]]

        dist = Poisson(dist_rates_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson_multivariate(self):
        dist_sample_shape_correct = [1, 3]
        dist_means_correct = [[1, 2, 15]]
        dist_stddevs_correct = [[math.sqrt(1), math.sqrt(2), math.sqrt(15)]]
        dist_rates_correct = [[1, 2, 15]]
        dist_log_probs_correct = [sum([-1, -1.30685, -2.27852])]

        dist = Poisson(dist_rates_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson_multivariate_from_flat_params(self):
        dist_sample_shape_correct = [1, 3]
        dist_means_correct = [[1, 2, 15]]
        dist_stddevs_correct = [[math.sqrt(1), math.sqrt(2), math.sqrt(15)]]
        dist_rates_correct = [[1, 2, 15]]
        dist_log_probs_correct = [sum([-1, -1.30685, -2.27852])]

        dist = Poisson(dist_rates_correct[0])
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_poisson_multivariate_batched(self):
        dist_sample_shape_correct = [2, 3]
        dist_means_correct = [[1, 2, 15], [100, 200, 300]]
        dist_stddevs_correct = [[math.sqrt(1), math.sqrt(2), math.sqrt(15)], [math.sqrt(100), math.sqrt(200), math.sqrt(300)]]
        dist_rates_correct = [[1, 2, 15], [100, 200, 300]]
        dist_log_probs_correct = [[sum([-1, -1.30685, -2.27852])], [sum([-3.22236, -3.56851, -3.77110])]]

        dist = Poisson(dist_rates_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_rates = util.to_numpy(dist.rate)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)

        dist_log_probs = util.to_numpy(dist.log_prob(dist_means_correct))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_rates', 'dist_rates_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_empirical_save_load(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        values = Variable(util.Tensor([1, 2, 3]))
        log_weights = Variable(util.Tensor([1, 2, 3]))
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


if __name__ == '__main__':
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
