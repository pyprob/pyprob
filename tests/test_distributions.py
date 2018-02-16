import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob import util
from pyprob.distributions import Categorical, Empirical, Normal, TruncatedNormal, Uniform


class DistributionsTestCase(unittest.TestCase):
    def test_dist_empirical(self):
        values = Variable(util.Tensor([1,2,3]))
        log_weights = Variable(util.Tensor([1,2,3]))
        dist_mean_correct = 2.5752103328704834
        dist_stddev_correct = 0.6514633893966675

        dist = Empirical(values, log_weights)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)

        util.debug('dist_mean', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)

    def test_dist_normal(self):
        empirical_samples = 10000
        dist_mean_correct = 2
        dist_stddev_correct = 3
        dist_log_prob_correct = -2.0175508218727822

        dist = Normal(dist_mean_correct, dist_stddev_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_log_prob = float(dist.log_prob(dist_mean_correct))

        util.debug('dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=1)

    def test_dist_truncated_normal(self):
        empirical_samples = 10000
        dist_mean_non_truncated_correct = 2
        dist_stddev_non_truncated_correct = 3
        dist_low_correct = -4
        dist_high_correct = 4
        dist_mean_correct = 0.9011890888214111
        dist_stddev_correct = 1.9511810541152954
        dist_log_prob_correct = -1.8206324577331543

        dist = TruncatedNormal(dist_mean_non_truncated_correct, dist_stddev_non_truncated_correct, dist_low_correct, dist_high_correct)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean_non_truncated = float(dist.mean_non_truncated)
        dist_stddev_non_truncated = float(dist.stddev_non_truncated)
        dist_low = float(dist.low)
        dist_high = float(dist.high)
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_log_prob = float(dist.log_prob(0.5))

        util.debug('dist_mean_non_truncated', 'dist_mean_non_truncated_correct', 'dist_stddev_non_truncated', 'dist_stddev_non_truncated_correct', 'dist_low', 'dist_low_correct', 'dist_high', 'dist_high_correct', 'dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_log_prob', 'dist_log_prob_correct')
        self.assertAlmostEqual(dist_mean_non_truncated, dist_mean_non_truncated_correct, places=1)
        self.assertAlmostEqual(dist_stddev_non_truncated, dist_stddev_non_truncated_correct, places=1)
        self.assertAlmostEqual(dist_low, dist_low_correct, places=1)
        self.assertAlmostEqual(dist_high, dist_high_correct, places=1)
        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=1)

    def test_dist_uniform(self):
        empirical_samples = 10000
        dist_mean_correct = 5
        dist_stddev_correct = 2.886751174926758
        dist_log_prob_correct = -2.3025851249694824

        dist = Uniform(0, 10)
        dist_empirical = Empirical([dist.sample() for i in range(empirical_samples)])
        dist_mean = float(dist.mean)
        dist_mean_empirical = float(dist_empirical.mean)
        dist_stddev = float(dist.stddev)
        dist_stddev_empirical = float(dist_empirical.stddev)
        dist_log_prob = float(dist.log_prob(5))

        util.debug('dist_mean', 'dist_mean_empirical', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_empirical', 'dist_stddev_correct', 'dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_mean_empirical, dist_mean_correct, places=1)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_stddev_empirical, dist_stddev_correct, places=1)
        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=1)

    def test_dist_categorical(self):
        dist_log_prob_correct = -0.6931471824645996

        dist = Categorical([0.2, 0.5, 0.2, 0.1])
        s = dist.sample()
        dist_log_prob = float(dist.log_prob(1))

        util.debug('dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=1)


if __name__ == '__main__':
    unittest.main()
