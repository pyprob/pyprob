import unittest
import math
import torch
import os
import tempfile
import uuid

import pyprob
from pyprob import util, Model, InferenceEngine
from pyprob.distributions import Normal, Uniform, Empirical



# http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
class GaussianWithUnknownMeanMarsaglia(Model):
    def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        self.likelihood_stddev = likelihood_stddev
        super().__init__('Gaussian with unknown mean (Marsaglia)')

    def marsaglia(self, mean, stddev):
        uniform = Uniform(-1, 1)
        s = 1
        while float(s) >= 1:
            x = pyprob.sample(uniform)
            y = pyprob.sample(uniform)
            s = x*x + y*y
        return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

    def forward(self):
        mu = self.marsaglia(self.prior_mean, self.prior_stddev)
        likelihood = Normal(mu, self.likelihood_stddev)
        pyprob.observe(likelihood, 0, name='obs0')
        pyprob.observe(likelihood, 0, name='obs1')
        return mu


model = GaussianWithUnknownMeanMarsaglia().parallel()


class ModelParallelTestCase(unittest.TestCase):
    def test_model_parallel_prior(self):
        num_traces = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = model.prior_results(num_traces)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.eval_print('num_traces', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_parallel_prior_on_disk(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        num_traces = 1000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)
        prior_length_correct = 2 * num_traces

        prior = model.prior_results(num_traces, file_name=file_name)
        prior.close()
        prior = model.prior_results(num_traces, file_name=file_name)
        # prior.close()
        prior_length = prior.length
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.eval_print('num_traces', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct', 'prior_length', 'prior_length_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)
        self.assertEqual(prior_length, prior_length_correct)

    def test_model_parallel_posterior_importance_sampling(self):
        samples = 2000
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.002

        posterior = model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, delta=0.75)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)


if __name__ == '__main__':
    pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
