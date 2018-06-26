import unittest
import torch
from torch.autograd import Variable
import numpy as np
import uuid
import tempfile
import os
import math

import pyprob
from pyprob import util, Model
from pyprob.distributions import Distribution, Categorical, Empirical, Mixture, Normal, TruncatedNormal, Uniform, Poisson, Kumaraswamy


empirical_samples = 20000


class DistributionsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self, observation=[]):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model_gum = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_dist_empirical(self):
        values = Variable(util.Tensor([1, 2, 3]))
        log_weights = Variable(util.Tensor([1, 2, 3]))
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
        dist_sample_shape = list(dist.sample().size())

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
        dist_combined_empirical = Empirical.combine([dist1_empirical, dist2_empirical, dist3_empirical])
        dist_combined_empirical_mean = float(dist_combined_empirical.mean)
        dist_combined_empirical_stddev = float(dist_combined_empirical.stddev)

        util.debug('dist1_empirical_mean', 'dist1_empirical_stddev', 'dist1_mean_correct', 'dist1_stddev_correct', 'dist2_empirical_mean', 'dist2_empirical_stddev', 'dist2_mean_correct', 'dist2_stddev_correct', 'dist3_empirical_mean', 'dist3_empirical_stddev', 'dist3_mean_correct', 'dist3_stddev_correct', 'dist_combined_empirical_mean', 'dist_combined_empirical_stddev', 'dist_combined_mean_correct', 'dist_combined_stddev_correct')

        self.assertAlmostEqual(dist1_empirical_mean, dist1_mean_correct, places=1)
        self.assertAlmostEqual(dist1_empirical_stddev, dist1_stddev_correct, places=1)
        self.assertAlmostEqual(dist2_empirical_mean, dist2_mean_correct, places=1)
        self.assertAlmostEqual(dist2_empirical_stddev, dist2_stddev_correct, places=1)
        self.assertAlmostEqual(dist3_empirical_mean, dist3_mean_correct, places=1)
        self.assertAlmostEqual(dist3_empirical_stddev, dist3_stddev_correct, places=1)
        self.assertAlmostEqual(dist_combined_empirical_mean, dist_combined_mean_correct, places=1)
        self.assertAlmostEqual(dist_combined_empirical_stddev, dist_combined_stddev_correct, places=1)

    def test_dist_empirical_combine_non_uniform_weights(self):
        samples1 = 10000
        samples2 = 1000
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior1 = self._model_gum.posterior_distribution(samples1, observation=observation)
        posterior1_mean = float(posterior1.mean)
        posterior1_mean_unweighted = float(posterior1.unweighted().mean)
        posterior1_stddev = float(posterior1.stddev)
        posterior1_stddev_unweighted = float(posterior1.unweighted().stddev)
        kl_divergence1 = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior1.mean, posterior1_stddev)))

        posterior2 = self._model_gum.posterior_distribution(samples2, observation=observation)
        posterior2_mean = float(posterior2.mean)
        posterior2_mean_unweighted = float(posterior2.unweighted().mean)
        posterior2_stddev = float(posterior2.stddev)
        posterior2_stddev_unweighted = float(posterior2.unweighted().stddev)
        kl_divergence2 = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior2.mean, posterior2_stddev)))

        posterior = Empirical.combine([posterior1, posterior2])
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples1', 'posterior1_mean_unweighted', 'posterior1_mean', 'posterior1_stddev_unweighted', 'posterior1_stddev', 'kl_divergence1', 'samples2', 'posterior2_mean_unweighted', 'posterior2_mean', 'posterior2_stddev_unweighted', 'posterior2_stddev', 'kl_divergence2', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior1_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior1_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence1, 0.25)
        self.assertAlmostEqual(posterior2_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior2_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence2, 0.25)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_dist_empirical_combine_non_uniform_weights_use_initial(self):
        samples1 = 10000
        samples2 = 1000
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior1 = self._model_gum.posterior_distribution(samples1, observation=observation)
        posterior1_mean = float(posterior1.mean)
        posterior1_mean_unweighted = float(posterior1.unweighted().mean)
        posterior1_stddev = float(posterior1.stddev)
        posterior1_stddev_unweighted = float(posterior1.unweighted().stddev)
        kl_divergence1 = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior1.mean, posterior1_stddev)))

        posterior2 = self._model_gum.posterior_distribution(samples2, observation=observation)
        posterior2_mean = float(posterior2.mean)
        posterior2_mean_unweighted = float(posterior2.unweighted().mean)
        posterior2_stddev = float(posterior2.stddev)
        posterior2_stddev_unweighted = float(posterior2.unweighted().stddev)
        kl_divergence2 = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior2.mean, posterior2_stddev)))

        posterior = Empirical.combine([posterior1, posterior2], use_initial_values_and_weights=True)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples1', 'posterior1_mean_unweighted', 'posterior1_mean', 'posterior1_stddev_unweighted', 'posterior1_stddev', 'kl_divergence1', 'samples2', 'posterior2_mean_unweighted', 'posterior2_mean', 'posterior2_stddev_unweighted', 'posterior2_stddev', 'kl_divergence2', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior1_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior1_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence1, 0.25)
        self.assertAlmostEqual(posterior2_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior2_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence2, 0.25)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

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

        dist = Mixture([dist_1, dist_2, dist_3], probs=[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])

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
        dist_sample_shape_correct = [4, 1]
        dist_means_correct = [[0.5], [7.5], [0.5], [0.5]]
        dist_stddevs_correct = [[0.288675], [1.44338], [0.288675], [0.288675]]
        dist_lows_correct = [[0], [5], [0], [0]]
        dist_highs_correct = [[1], [10], [1], [1]]
        dist_values = [[0.5], [7.5], [0], [1]]
        dist_log_probs_correct = [[0], [-1.60944], [float('-inf')], [float('-inf')]]

        dist = Uniform(dist_lows_correct, dist_highs_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_lows = util.to_numpy(dist.low)
        dist_highs = util.to_numpy(dist.high)
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_values))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_lows', 'dist_lows_correct', 'dist_highs', 'dist_highs_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_values', 'dist_log_probs', 'dist_log_probs_correct')

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
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.25))
        self.assertTrue(np.allclose(dist_rates, dist_rates_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_log_probs, dist_log_probs_correct, atol=0.1))

    def test_dist_kumaraswamy(self):
        dist_sample_shape_correct = [1]
        dist_shape1s_correct = [2]
        dist_shape2s_correct = [5]
        dist_means_correct = [0.369408]
        dist_stddevs_correct = [0.173793]
        dist_log_probs_correct = [0.719861]

        dist = Kumaraswamy(dist_shape1s_correct, dist_shape2s_correct)
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

    def test_dist_kumaraswamy_batched(self):
        dist_sample_shape_correct = [4, 1]
        dist_shape1s_correct = [[0.5], [7.5], [7.5], [7.5]]
        dist_shape2s_correct = [[0.75], [2.5], [2.5], [2.5]]
        dist_means_correct = [[0.415584], [0.807999], [0.807999], [0.807999]]
        dist_stddevs_correct = [[0.327509], [0.111605], [0.111605], [0.111605]]
        dist_values = [[0.415584], [0.807999], [0.], [1.]]
        dist_log_probs_correct = [[-0.283125], [1.20676], [float('-inf')], [float('-inf')]]

        dist = Kumaraswamy(dist_shape1s_correct, dist_shape2s_correct)
        dist_sample_shape = list(dist.sample().size())
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_means = util.to_numpy(dist.mean)
        dist_means_empirical = util.to_numpy(dist_empirical.mean)
        dist_stddevs = util.to_numpy(dist.stddev)
        dist_stddevs_empirical = util.to_numpy(dist_empirical.stddev)
        dist_log_probs = util.to_numpy(dist.log_prob(dist_values))

        util.debug('dist_sample_shape', 'dist_sample_shape_correct', 'dist_means', 'dist_means_empirical', 'dist_means_correct', 'dist_stddevs', 'dist_stddevs_empirical', 'dist_stddevs_correct', 'dist_values', 'dist_log_probs', 'dist_log_probs_correct')

        self.assertEqual(dist_sample_shape, dist_sample_shape_correct)
        self.assertTrue(np.allclose(dist_means, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_means_empirical, dist_means_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs, dist_stddevs_correct, atol=0.1))
        self.assertTrue(np.allclose(dist_stddevs_empirical, dist_stddevs_correct, atol=0.1))
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
