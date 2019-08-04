import unittest
import math
import torch

import pyprob
from pyprob import util, Model
from pyprob.distributions import Normal, Uniform
import pyprob.diagnostics


class DiagnosticsTestCase(unittest.TestCase):
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

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, name='obs0')
                pyprob.observe(likelihood, name='obs1')
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_prior_statistics(self):
        num_traces = 2000
        trace_length_mean_correct = 4.543580055236816  # Reference value from 100k runs
        trace_length_stddev_correct = 1.177796721458435  # Reference value from 100k runs
        trace_length_min_correct = 4

        stats = pyprob.diagnostics._trace_stats(self._model.prior(num_traces=num_traces))
        trace_length_mean = stats['traces_extra']['trace_length_mean']
        trace_length_stddev = stats['traces_extra']['trace_length_stddev']
        trace_length_min = stats['traces_extra']['trace_length_min']
        trace_length_max = stats['traces_extra']['trace_length_max']

        util.eval_print('num_traces', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)

    def test_posterior_statistics(self):
        num_traces = 2000
        trace_length_mean_correct = 4.556660175323486  # Reference value from 100k runs
        trace_length_stddev_correct = 1.1909255981445312  # Reference value from 100k runs
        trace_length_min_correct = 4

        stats = pyprob.diagnostics._trace_stats(self._model.posterior(num_traces=num_traces, observe={'obs0': 8, 'obs1': 9}))
        trace_length_mean = stats['traces_extra']['trace_length_mean']
        trace_length_stddev = stats['traces_extra']['trace_length_stddev']
        trace_length_min = stats['traces_extra']['trace_length_min']
        trace_length_max = stats['traces_extra']['trace_length_max']

        util.eval_print('num_traces', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
