import unittest
import math
import torch
import uuid
import tempfile
import os

import pyprob
from pyprob import util, Model, InferenceEngine
from pyprob.distributions import Normal, Uniform


class ModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
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

            def forward(self, observation=[]):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_model_prior(self):
        samples = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_trace_length_statistics(self):
        samples = 2000
        trace_length_mean_correct = 2.5630438327789307
        trace_length_stddev_correct = 1.2081329822540283
        trace_length_min_correct = 2

        self._model._trace_statistics_samples = samples
        trace_length_mean = float(self._model.trace_length_mean(samples))
        trace_length_stddev = float(self._model.trace_length_stddev(samples))
        trace_length_min = float(self._model.trace_length_min(samples))
        trace_length_max = float(self._model.trace_length_max(samples))

        util.debug('samples', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)

    def test_model_train_save_load(self):
        training_traces = 128
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        self._model.learn_inference_network(observation=[1, 1], num_traces=training_traces)
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)

        util.debug('training_traces', 'file_name')

        self.assertTrue(True)

    def test_model_lmh_posterior_with_initial_trace(self):
        num_traces = 256

        trace = next(self._model._prior_trace_generator())
        self._model.posterior_distribution(num_traces=num_traces, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, initial_trace=trace)

        self.assertTrue(True)

class ModelWithReplacementTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMeanMarsagliaWithReplacement(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean (Marsaglia), with replacement')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform, replace=True)
                    y = pyprob.sample(uniform, replace=True)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self, observation=[]):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMeanMarsagliaWithReplacement()
        super().__init__(*args, **kwargs)

    def test_model_with_replacement_trace_length_statistics(self):
        samples = 2000
        trace_length_mean_correct = 2
        trace_length_stddev_correct = 0
        trace_length_min_correct = 2
        trace_length_max_correct = 2

        self._model._trace_statistics_samples = samples
        trace_length_mean = float(self._model.trace_length_mean(samples))
        trace_length_stddev = float(self._model.trace_length_stddev(samples))
        trace_length_min = float(self._model.trace_length_min(samples))
        trace_length_max = float(self._model.trace_length_max(samples))

        util.debug('samples', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max', 'trace_length_max_correct')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)
        self.assertAlmostEqual(trace_length_max, trace_length_max_correct, places=0)


if __name__ == '__main__':
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
