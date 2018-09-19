import unittest
import math
import torch
import os
import tempfile
import uuid

import pyprob
from pyprob import util, Model, InferenceEngine
from pyprob.distributions import Normal, Uniform, Empirical


importance_sampling_samples = 5000


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

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, 0, name='obs0')
                pyprob.observe(likelihood, 0, name='obs1')
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_model_prior(self):
        num_traces = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(num_traces)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('num_traces', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_trace_length_statistics(self):
        num_traces = 2000
        trace_length_mean_correct = 2.5630438327789307
        trace_length_stddev_correct = 1.2081329822540283
        trace_length_min_correct = 2

        trace_lengths = self._model.prior_traces(num_traces, map_func=lambda trace: trace.length_controlled)
        trace_length_dist = Empirical(trace_lengths)
        trace_length_mean = float(trace_length_dist.mean)
        trace_length_stddev = float(trace_length_dist.stddev)
        trace_length_min = float(trace_length_dist.min)
        trace_length_max = (trace_length_dist.max)

        util.debug('num_traces', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)

    def test_model_train_save_load_train(self):
        training_traces = 128
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        self._model.learn_inference_network(num_traces=training_traces, observe_embeddings={'obs0': {'dim': 64}, 'obs1': {'dim': 64}})
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)
        self._model.learn_inference_network(num_traces=training_traces, observe_embeddings={'obs0': {'dim': 64}, 'obs1': {'dim': 64}})

        util.debug('training_traces', 'file_name')

        self.assertTrue(True)

    def test_model_lmh_posterior_with_stop_and_resume(self):
        posterior_num_runs = 100
        posterior_num_traces_each_run = 20
        posterior_num_traces_correct = posterior_num_traces_each_run * posterior_num_runs
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)

        posteriors = []
        initial_trace = None
        for i in range(posterior_num_runs):
            posterior = self._model.posterior_traces(num_traces=posterior_num_traces_each_run, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9}, initial_trace=initial_trace)
            initial_trace = posterior[-1]
            posteriors.append(posterior)
        posterior = Empirical.combine(posteriors).map(lambda trace: trace.result)
        posterior_num_traces = posterior.length
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('posterior_num_runs', 'posterior_num_traces_each_run', 'posterior_num_traces', 'posterior_num_traces_correct', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertEqual(posterior_num_traces, posterior_num_traces_correct)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_model_rmh_posterior_with_stop_and_resume(self):
        posterior_num_runs = 100
        posterior_num_traces_each_run = 20
        posterior_num_traces_correct = posterior_num_traces_each_run * posterior_num_runs
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)

        posteriors = []
        initial_trace = None
        for i in range(posterior_num_runs):
            posterior = self._model.posterior_traces(num_traces=posterior_num_traces_each_run, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9}, initial_trace=initial_trace)
            initial_trace = posterior[-1]
            posteriors.append(posterior)
        posterior = Empirical.combine(posteriors).map(lambda trace: trace.result)
        posterior_num_traces = posterior.length
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('posterior_num_runs', 'posterior_num_traces_each_run', 'posterior_num_traces', 'posterior_num_traces_correct', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertEqual(posterior_num_traces, posterior_num_traces_correct)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class ModelWithReplacementTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMeanMarsagliaWithReplacement(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean (Marsaglia)')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform, replace=True)
                    y = pyprob.sample(uniform, replace=True)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, 0, name='obs0')
                pyprob.observe(likelihood, 0, name='obs1')
                return mu

        self._model = GaussianWithUnknownMeanMarsagliaWithReplacement()
        super().__init__(*args, **kwargs)

    def test_model_with_replacement_trace_length_statistics(self):
        num_traces = 2000
        trace_length_mean_correct = 2
        trace_length_stddev_correct = 0
        trace_length_min_correct = 2
        trace_length_max_correct = 2

        trace_lengths = self._model.prior_traces(num_traces, map_func=lambda trace: trace.length_controlled)
        trace_length_dist = Empirical(trace_lengths)
        trace_length_mean = float(trace_length_dist.mean)
        trace_length_stddev = float(trace_length_dist.stddev)
        trace_length_min = float(trace_length_dist.min)
        trace_length_max = (trace_length_dist.max)

        util.debug('num_traces', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max', 'trace_length_max_correct')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)
        self.assertAlmostEqual(trace_length_max, trace_length_max_correct, places=0)


class ModelObservationStyle1TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                # pyprob.observe usage alternative #1
                pyprob.observe(likelihood, name='obs0')
                pyprob.observe(likelihood, name='obs1')
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_observation_style1_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)

        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class ModelObservationStyle2TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                # pyprob.observe usage alternative #2
                pyprob.sample(likelihood, name='obs0')
                pyprob.sample(likelihood, name='obs1')
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_observation_style2_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)

        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(2)
    unittest.main(verbosity=2)
