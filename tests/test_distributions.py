import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob import util
from pyprob.distributions import Empirical, Normal


class TestCase(unittest.TestCase):
    def test_dist_empirical(self):
        values = Variable(util.Tensor([1,2,3]))
        log_weights = Variable(util.Tensor([1,2,3]))
        dist = Empirical(values, log_weights)
        s = dist.sample()
        print(dist)
        dist_mean = float(dist.mean)
        correct_dist_mean = 2.5752103328704834
        util.debug('dist_mean', 'correct_dist_mean')
        self.assertAlmostEqual(dist_mean, correct_dist_mean, places=0)
        dist_stddev = float(dist.stddev)
        correct_dist_stddev = 0.6514633893966675
        util.debug('dist_stddev', 'correct_dist_stddev')
        self.assertAlmostEqual(dist_stddev, correct_dist_stddev, places=0)

    def test_dist_normal(self):
        dist = Normal(2, 3)
        s = dist.sample()
        dist_mean = float(dist.mean)
        correct_dist_mean = 2
        util.debug('dist_mean', 'correct_dist_mean')
        self.assertAlmostEqual(dist_mean, correct_dist_mean, places=0)
        dist_stddev = float(dist.stddev)
        correct_dist_stddev = 3
        util.debug('dist_stddev', 'correct_dist_stddev')
        self.assertAlmostEqual(dist_stddev, correct_dist_stddev, places=0)
        logprob = float(dist.log_prob(2))
        correct_logprob = -2.0175508218727822
        util.debug('logprob', 'correct_logprob')
        self.assertAlmostEqual(logprob, correct_logprob, places=0)
