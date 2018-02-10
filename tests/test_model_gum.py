import unittest
import math
import numpy as np

import pyprob
from pyprob import util
from pyprob import Model
from pyprob.distributions import Empirical, Normal


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('GaussianWithUnknownMean')

            def forward(self, obs=[]):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in obs:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_model_gum_prior(self):
        prior = self._model.prior_distribution(5000)
        prior_mean = float(prior.mean)
        correct_prior_mean = 1
        util.debug('prior_mean', 'correct_prior_mean')
        self.assertAlmostEqual(prior_mean, correct_prior_mean, places=0)
        prior_stddev = float(prior.stddev)
        correct_prior_stddev = math.sqrt(5)
        util.debug('prior_stddev', 'correct_prior_stddev')
        self.assertAlmostEqual(prior_stddev, correct_prior_stddev, places=0)

    def test_model_gum_posterior_importance_sampling(self):
        posterior = self._model.posterior_distribution(5000, obs=[8,9])
        posterior_mean = float(posterior.mean)
        correct_posterior_mean = 7.25
        util.debug('posterior_mean', 'correct_posterior_mean')
        self.assertAlmostEqual(posterior_mean, correct_posterior_mean, places=0)
        posterior_stddev = float(posterior.stddev)
        correct_posterior_stddev = math.sqrt(1/1.2)
        util.debug('posterior_stddev', 'correct_posterior_stddev')
        self.assertAlmostEqual(posterior_stddev, correct_posterior_stddev, places=0)

    def test_model_gum_posterior_inference_compilation(self):
        self.assertTrue(True)
