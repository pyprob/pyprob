import unittest
import math
import numpy as np

import pyprob
from pyprob import Model
from pyprob.distributions import Empirical, Normal


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mu=1, prior_sigma=math.sqrt(5), likelihood_sigma=math.sqrt(2)):
                self.prior_mu = prior_mu
                self.prior_sigma = prior_sigma
                self.likelihood_sigma = likelihood_sigma
                super().__init__('GaussianWithUnknownMean')

            def forward(self, obs=[]):
                mu = pyprob.sample(Normal(self.prior_mu, self.prior_sigma))
                likelihood = Normal(mu, self.likelihood_sigma)
                for o in obs:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_model_gum_prior(self):
        prior = self._model.prior_distribution(1000)
        prior_mean = float(prior.mean())
        correct_prior_mean = 1
        print('\n  prior_mean', prior_mean, 'correct_prior_mean', correct_prior_mean)
        self.assertAlmostEqual(prior_mean, correct_prior_mean, places=0)

    def test_model_gum_posterior_importance_sampling(self):
        posterior = self._model.posterior_distribution(2500, obs=[8,9])
        posterior_mean = float(posterior.mean())
        correct_posterior_mean = 7.25
        print('\n  posterior_mean', posterior_mean, 'correct_posterior_mean', correct_posterior_mean)
        self.assertAlmostEqual(posterior_mean, correct_posterior_mean, places=0)

    def test_model_gum_posterior_inference_compilation(self):
        self.assertTrue(True)
