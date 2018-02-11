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

            def forward(self, observation=[]):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_model_gum_prior(self):
        samples = 5000
        correct_prior_mean = 1
        correct_prior_stddev = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'correct_prior_mean', 'prior_stddev', 'correct_prior_stddev')

        self.assertAlmostEqual(prior_mean, correct_prior_mean, places=0)
        self.assertAlmostEqual(prior_stddev, correct_prior_stddev, places=0)

    def test_model_gum_posterior_importance_sampling(self):
        samples = 5000
        correct_posterior_mean = 7.25
        correct_posterior_stddev = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=[8,9])
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'correct_posterior_mean', 'posterior_stddev_unweighted', 'posterior_stddev', 'correct_posterior_stddev')

        self.assertAlmostEqual(posterior_mean, correct_posterior_mean, places=0)
        self.assertAlmostEqual(posterior_stddev, correct_posterior_stddev, places=0)

    def test_model_gum_posterior_inference_compilation(self):
        training_traces = 5000
        samples = 500
        correct_posterior_mean = 7.25
        correct_posterior_stddev = math.sqrt(1/1.2)

        self._model.learn_proposal(observation=[1,1], max_traces=training_traces)
        posterior = self._model.posterior_distribution(samples, learned_proposal=True, observation=[8,9])
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'correct_posterior_mean', 'posterior_stddev_unweighted', 'posterior_stddev', 'correct_posterior_stddev')

        self.assertAlmostEqual(posterior_mean, correct_posterior_mean, places=0)
        self.assertAlmostEqual(posterior_stddev, correct_posterior_stddev, places=0)
