import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob import util
from pyprob.distributions import Empirical, Normal, Uniform, Categorical


class TestCase(unittest.TestCase):
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

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=0)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=0)

    def test_dist_normal(self):
        dist_mean_correct = 2
        dist_stddev_correct = 3
        dist_log_prob_correct = -2.0175508218727822

        dist = Normal(dist_mean_correct, dist_stddev_correct)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)
        dist_log_prob = float(dist.log_prob(dist_mean_correct))

        util.debug('dist_mean', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_correct', 'dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=0)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=0)
        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=0)

    def test_dist_uniform(self):
        dist_mean_correct = 5
        dist_stddev_correct = 2.886751174926758
        dist_log_prob_correct = -2.3025851249694824

        dist = Uniform(0, 10)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)
        dist_log_prob = float(dist.log_prob(5))

        util.debug('dist_mean', 'dist_mean_correct', 'dist_stddev', 'dist_stddev_correct', 'dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_mean, dist_mean_correct, places=0)
        self.assertAlmostEqual(dist_stddev, dist_stddev_correct, places=0)
        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=0)

    def test_dist_categorical(self):
        dist_log_prob_correct = -0.6931471824645996

        dist = Categorical([0.2, 0.5, 0.2, 0.1])
        s = dist.sample()
        dist_log_prob = float(dist.log_prob(1))

        util.debug('dist_log_prob', 'dist_log_prob_correct')

        self.assertAlmostEqual(dist_log_prob, dist_log_prob_correct, places=0)
