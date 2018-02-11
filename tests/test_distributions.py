import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob import util
from pyprob.distributions import Empirical, Normal, Uniform


class TestCase(unittest.TestCase):
    def test_dist_empirical(self):
        values = Variable(util.Tensor([1,2,3]))
        log_weights = Variable(util.Tensor([1,2,3]))
        correct_dist_mean = 2.5752103328704834
        correct_dist_stddev = 0.6514633893966675

        dist = Empirical(values, log_weights)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)

        util.debug('dist_mean', 'correct_dist_mean', 'dist_stddev', 'correct_dist_stddev')

        self.assertAlmostEqual(dist_mean, correct_dist_mean, places=0)
        self.assertAlmostEqual(dist_stddev, correct_dist_stddev, places=0)

    def test_dist_normal(self):
        correct_dist_mean = 2
        correct_dist_stddev = 3
        correct_logprob = -2.0175508218727822

        dist = Normal(correct_dist_mean, correct_dist_stddev)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)
        logprob = float(dist.log_prob(correct_dist_mean))

        util.debug('dist_mean', 'correct_dist_mean', 'dist_stddev', 'correct_dist_stddev', 'logprob', 'correct_logprob')

        self.assertAlmostEqual(dist_mean, correct_dist_mean, places=0)
        self.assertAlmostEqual(dist_stddev, correct_dist_stddev, places=0)
        self.assertAlmostEqual(logprob, correct_logprob, places=0)

    def test_dist_uniform(self):
        correct_dist_mean = 5
        correct_dist_stddev = 2.886751174926758
        correct_logprob = -2.3025851249694824

        dist = Uniform(0, 10)
        s = dist.sample()
        dist_mean = float(dist.mean)
        dist_stddev = float(dist.stddev)
        logprob = float(dist.log_prob(5))

        util.debug('dist_mean', 'correct_dist_mean', 'dist_stddev', 'correct_dist_stddev', 'logprob', 'correct_logprob')

        self.assertAlmostEqual(dist_mean, correct_dist_mean, places=0)
        self.assertAlmostEqual(dist_stddev, correct_dist_stddev, places=0)
        self.assertAlmostEqual(logprob, correct_logprob, places=0)
