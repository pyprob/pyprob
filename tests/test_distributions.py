import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob.distributions import Empirical, Normal


class TestCase(unittest.TestCase):
    def test_dist_empirical(self):
        values = Variable(torch.Tensor([1,2,3]))
        log_weights = Variable(torch.Tensor([2,2,2]))
        dist = Empirical(values, log_weights)
        s = dist.sample()
        self.assertTrue(True)

    def test_dist_normal(self):
        dist = Normal(2, 3)
        s = dist.sample()
        logp = dist.log_prob(s)
        self.assertTrue(True)
