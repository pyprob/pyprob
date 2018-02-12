import unittest
import math
import torch

import pyprob
from pyprob import util
from pyprob import Model
from pyprob.distributions import Empirical, Normal, Uniform


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMeanMarsaglia(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('GaussianWithUnknownMeanMarsaglia')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform)
                    y = pyprob.sample(uniform)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self, observation=[]):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_model_gum_marsaglia_prior(self):
        samples = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_inference_gum_marsaglia_posterior_importance_sampling(self):
        samples = 5000
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.1)
