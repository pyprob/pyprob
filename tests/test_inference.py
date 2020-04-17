# import unittest
# import torch
# import torch.nn.functional as F
# import numpy as np
# import time
# import math
# import sys
# import functools
# from PIL import Image, ImageDraw, ImageFont
# from termcolor import colored
#
# import pyprob
# from pyprob import util, Model, InferenceEngine, InferenceNetwork, PriorInflation, ObserveEmbedding
# from pyprob.distributions import Normal, Uniform, Categorical, Poisson, Empirical
#
#
# importance_sampling_samples = 5000
# importance_sampling_kl_divergence = 0
# importance_sampling_duration = 0
#
# importance_sampling_with_inference_network_ff_samples = 5000
# importance_sampling_with_inference_network_ff_kl_divergence = 0
# importance_sampling_with_inference_network_ff_duration = 0
# importance_sampling_with_inference_network_ff_training_traces = 50000
# importance_sampling_with_inference_network_ff_prior_inflation = PriorInflation.ENABLED
#
# importance_sampling_with_inference_network_lstm_samples = 5000
# importance_sampling_with_inference_network_lstm_kl_divergence = 0
# importance_sampling_with_inference_network_lstm_duration = 0
# importance_sampling_with_inference_network_lstm_training_traces = 50000
# importance_sampling_with_inference_network_lstm_prior_inflation = PriorInflation.ENABLED
#
# lightweight_metropolis_hastings_samples = 5000
# lightweight_metropolis_hastings_burn_in = 500
# lightweight_metropolis_hastings_kl_divergence = 0
# lightweight_metropolis_hastings_duration = 0
#
# random_walk_metropolis_hastings_samples = 5000
# random_walk_metropolis_hastings_burn_in = 500
# random_walk_metropolis_hastings_kl_divergence = 0
# random_walk_metropolis_hastings_duration = 0
#
#
# def add_importance_sampling_kl_divergence(val):
#     global importance_sampling_kl_divergence
#     importance_sampling_kl_divergence += val
#
#
# def add_importance_sampling_with_inference_network_ff_kl_divergence(val):
#     global importance_sampling_with_inference_network_ff_kl_divergence
#     importance_sampling_with_inference_network_ff_kl_divergence += val
#
#
# def add_importance_sampling_with_inference_network_lstm_kl_divergence(val):
#     global importance_sampling_with_inference_network_lstm_kl_divergence
#     importance_sampling_with_inference_network_lstm_kl_divergence += val
#
#
# def add_lightweight_metropolis_hastings_kl_divergence(val):
#     global lightweight_metropolis_hastings_kl_divergence
#     lightweight_metropolis_hastings_kl_divergence += val
#
#
# def add_random_walk_metropolis_hastings_kl_divergence(val):
#     global random_walk_metropolis_hastings_kl_divergence
#     random_walk_metropolis_hastings_kl_divergence += val
#
#
# def add_importance_sampling_duration(val):
#     global importance_sampling_duration
#     importance_sampling_duration += val
#
#
# def add_importance_sampling_with_inference_network_ff_duration(val):
#     global importance_sampling_with_inference_network_ff_duration
#     importance_sampling_with_inference_network_ff_duration += val
#
#
# def add_importance_sampling_with_inference_network_lstm_duration(val):
#     global importance_sampling_with_inference_network_lstm_duration
#     importance_sampling_with_inference_network_lstm_duration += val
#
#
# def add_lightweight_metropolis_hastings_duration(val):
#     global lightweight_metropolis_hastings_duration
#     lightweight_metropolis_hastings_duration += val
#
#
# def add_random_walk_metropolis_hastings_duration(val):
#     global random_walk_metropolis_hastings_duration
#     random_walk_metropolis_hastings_duration += val
#
#
# class GaussianWithUnknownMeanTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
#         class GaussianWithUnknownMean(Model):
#             def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
#                 self.prior_mean = prior_mean
#                 self.prior_stddev = prior_stddev
#                 self.likelihood_stddev = likelihood_stddev
#                 super().__init__('Gaussian with unknown mean')
#
#             def forward(self):
#                 mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
#                 likelihood = Normal(mu, self.likelihood_stddev)
#                 pyprob.observe(likelihood, name='obs0')
#                 pyprob.observe(likelihood, name='obs1')
#                 return mu
#
#         self._model = GaussianWithUnknownMean()
#         super().__init__(*args, **kwargs)
#
#     def test_inference_gum_posterior_importance_sampling(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         prior_mean_correct = 1.
#         prior_stddev_correct = math.sqrt(5)
#         posterior_effective_sample_size_min = samples * 0.005
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_posterior_importance_sampling_with_inference_network_ff(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         posterior_effective_sample_size_min = samples * 0.2
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_ff_training_traces, observe_embeddings={'obs0': {'dim': 128, 'depth': 6}, 'obs1': {'dim': 128, 'depth': 6}}, prior_inflation=importance_sampling_with_inference_network_ff_prior_inflation, inference_network=InferenceNetwork.FEEDFORWARD)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_with_inference_network_ff_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_ff_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_posterior_importance_sampling_with_inference_network_lstm(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         posterior_effective_sample_size_min = samples * 0.2
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_lstm_training_traces, observe_embeddings={'obs0': {'dim': 64, 'depth': 6}, 'obs1': {'dim': 64, 'depth': 6}}, prior_inflation=importance_sampling_with_inference_network_lstm_prior_inflation, inference_network=InferenceNetwork.LSTM)
#
#         # pyprob.diagnostics.network_statistics(self._model._inference_network, './report_tmp')
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_with_inference_network_lstm_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_lstm_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_posterior_lightweight_metropolis_hastings(self):
#         samples = lightweight_metropolis_hastings_samples
#         burn_in = lightweight_metropolis_hastings_burn_in
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
#         add_lightweight_metropolis_hastings_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_stddev = float(posterior.stddev)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
#         add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_posterior_random_walk_metropolis_hastings(self):
#         samples = random_walk_metropolis_hastings_samples
#         burn_in = random_walk_metropolis_hastings_burn_in
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
#         add_random_walk_metropolis_hastings_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_stddev = float(posterior.stddev)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
#         add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertLess(kl_divergence, 0.25)
#
#
# class GaussianWithUnknownMeanMarsagliaTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
#         class GaussianWithUnknownMeanMarsaglia(Model):
#             def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
#                 self.prior_mean = prior_mean
#                 self.prior_stddev = prior_stddev
#                 self.likelihood_stddev = likelihood_stddev
#                 super().__init__('Gaussian with unknown mean (Marsaglia)')
#
#             def marsaglia(self, mean, stddev):
#                 uniform = Uniform(-1, 1)
#                 s = 1
#                 while float(s) >= 1:
#                     x = pyprob.sample(uniform, replace=True)
#                     y = pyprob.sample(uniform, replace=True)
#                     s = x*x + y*y
#                 return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))
#
#             def forward(self):
#                 mu = self.marsaglia(self.prior_mean, self.prior_stddev)
#                 likelihood = Normal(mu, self.likelihood_stddev)
#                 pyprob.observe(likelihood, name='obs0')
#                 pyprob.observe(likelihood, name='obs1')
#                 return mu
#
#         self._model = GaussianWithUnknownMeanMarsaglia()
#         super().__init__(*args, **kwargs)
#
#     def test_inference_gum_marsaglia_posterior_importance_sampling(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         prior_mean_correct = 1.
#         prior_stddev_correct = math.sqrt(5)
#         posterior_effective_sample_size_min = samples * 0.002
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_marsaglia_posterior_importance_sampling_with_inference_network_ff(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         posterior_effective_sample_size_min = samples * 0.01
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_ff_training_traces, observe_embeddings={'obs0': {'dim': 128, 'depth': 6}, 'obs1': {'dim': 128, 'depth': 6}}, prior_inflation=importance_sampling_with_inference_network_ff_prior_inflation, inference_network=InferenceNetwork.FEEDFORWARD)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_with_inference_network_ff_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_ff_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_marsaglia_posterior_importance_sampling_with_inference_network_lstm(self):
#         samples = importance_sampling_samples
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#         posterior_effective_sample_size_min = samples * 0.02
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_ff_training_traces, observe_embeddings={'obs0': {'dim': 128, 'depth': 6}, 'obs1': {'dim': 128, 'depth': 6}}, prior_inflation=importance_sampling_with_inference_network_lstm_prior_inflation, inference_network=InferenceNetwork.LSTM)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
#         add_importance_sampling_with_inference_network_lstm_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_mean_unweighted = float(posterior.unweighted().mean)
#         posterior_stddev = float(posterior.stddev)
#         posterior_stddev_unweighted = float(posterior.unweighted().stddev)
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_lstm_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_marsaglia_posterior_lightweight_metropolis_hastings(self):
#         samples = lightweight_metropolis_hastings_samples
#         burn_in = lightweight_metropolis_hastings_burn_in
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
#         add_lightweight_metropolis_hastings_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_stddev = float(posterior.stddev)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
#         add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_gum_marsaglia_posterior_random_walk_metropolis_hastings(self):
#         samples = random_walk_metropolis_hastings_samples
#         burn_in = random_walk_metropolis_hastings_burn_in
#         true_posterior = Normal(7.25, math.sqrt(1/1.2))
#         posterior_mean_correct = float(true_posterior.mean)
#         posterior_stddev_correct = float(true_posterior.stddev)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
#         add_random_walk_metropolis_hastings_duration(time.time() - start)
#
#         posterior_mean = float(posterior.mean)
#         posterior_stddev = float(posterior.stddev)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
#         add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
#         self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
#         self.assertLess(kl_divergence, 0.25)
#
#
# class HiddenMarkovModelTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
#         class HiddenMarkovModel(Model):
#             def __init__(self, init_dist, trans_dists, obs_dists, obs_length):
#                 self.init_dist = init_dist
#                 self.trans_dists = trans_dists
#                 self.obs_dists = obs_dists
#                 self.obs_length = obs_length
#                 super().__init__('Hidden Markov model')
#
#             def forward(self):
#                 states = [pyprob.sample(init_dist)]
#                 for i in range(self.obs_length):
#                     state = pyprob.sample(self.trans_dists[int(states[-1])])
#                     pyprob.observe(self.obs_dists[int(state)], name='obs{}'.format(i))
#                     states.append(state)
#                 return torch.stack([util.one_hot(3, int(s)) for s in states])
#
#         init_dist = Categorical([1, 1, 1])
#         trans_dists = [Categorical([0.1, 0.5, 0.4]),
#                        Categorical([0.2, 0.2, 0.6]),
#                        Categorical([0.15, 0.15, 0.7])]
#         obs_dists = [Normal(-1, 1),
#                      Normal(1, 1),
#                      Normal(0, 1)]
#
#         self._observation = [0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]
#         self._model = HiddenMarkovModel(init_dist, trans_dists, obs_dists, len(self._observation))
#         self._posterior_mean_correct = util.to_tensor([[0.3775, 0.3092, 0.3133],
#                                                        [0.0416, 0.4045, 0.5539],
#                                                        [0.0541, 0.2552, 0.6907],
#                                                        [0.0455, 0.2301, 0.7244],
#                                                        [0.1062, 0.1217, 0.7721],
#                                                        [0.0714, 0.1732, 0.7554],
#                                                        [0.9300, 0.0001, 0.0699],
#                                                        [0.4577, 0.0452, 0.4971],
#                                                        [0.0926, 0.2169, 0.6905],
#                                                        [0.1014, 0.1359, 0.7626],
#                                                        [0.0985, 0.1575, 0.7440],
#                                                        [0.1781, 0.2198, 0.6022],
#                                                        [0.0000, 0.9848, 0.0152],
#                                                        [0.1130, 0.1674, 0.7195],
#                                                        [0.0557, 0.1848, 0.7595],
#                                                        [0.2017, 0.0472, 0.7511],
#                                                        [0.2545, 0.0611, 0.6844]])
#         super().__init__(*args, **kwargs)
#
#     def test_inference_hmm_posterior_importance_sampling(self):
#         samples = importance_sampling_samples
#         observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
#         posterior_mean_correct = self._posterior_mean_correct
#         posterior_effective_sample_size_min = samples * 0.001
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, observe=observation)
#         add_importance_sampling_duration(time.time() - start)
#         posterior_mean_unweighted = posterior.unweighted().mean
#         posterior_mean = posterior.mean
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#
#         l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
#         add_importance_sampling_kl_divergence(kl_divergence)
#
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(l2_distance, 3)
#         self.assertLess(kl_divergence, 1)
#
#     def test_inference_hmm_posterior_importance_sampling_with_inference_network_ff(self):
#         samples = importance_sampling_with_inference_network_ff_samples
#         observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
#         posterior_mean_correct = self._posterior_mean_correct
#         posterior_effective_sample_size_min = samples * 0.001
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_ff_training_traces, observe_embeddings={'obs{}'.format(i): {'depth': 2, 'dim': 32} for i in range(len(observation))}, prior_inflation=importance_sampling_with_inference_network_ff_prior_inflation, inference_network=InferenceNetwork.FEEDFORWARD)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe=observation)
#         add_importance_sampling_with_inference_network_ff_duration(time.time() - start)
#         posterior_mean_unweighted = posterior.unweighted().mean
#         posterior_mean = posterior.mean
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#
#         l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
#         add_importance_sampling_with_inference_network_ff_kl_divergence(kl_divergence)
#
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(l2_distance, 3)
#         self.assertLess(kl_divergence, 1)
#
#     def test_inference_hmm_posterior_importance_sampling_with_inference_network_lstm(self):
#         samples = importance_sampling_with_inference_network_ff_samples
#         observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
#         posterior_mean_correct = self._posterior_mean_correct
#         posterior_effective_sample_size_min = samples * 0.001
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_lstm_training_traces, observe_embeddings={'obs{}'.format(i): {'depth': 2, 'dim': 32} for i in range(len(observation))}, prior_inflation=importance_sampling_with_inference_network_lstm_prior_inflation, inference_network=InferenceNetwork.LSTM)
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe=observation)
#         add_importance_sampling_with_inference_network_lstm_duration(time.time() - start)
#         posterior_mean_unweighted = posterior.unweighted().mean
#         posterior_mean = posterior.mean
#         posterior_effective_sample_size = float(posterior.effective_sample_size)
#
#         l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))
#
#         util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
#         add_importance_sampling_with_inference_network_lstm_kl_divergence(kl_divergence)
#
#         self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
#         self.assertLess(l2_distance, 3)
#         self.assertLess(kl_divergence, 1)
#
#     def test_inference_hmm_posterior_lightweight_metropolis_hastings(self):
#         samples = lightweight_metropolis_hastings_samples
#         burn_in = lightweight_metropolis_hastings_burn_in
#         observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
#         posterior_mean_correct = self._posterior_mean_correct
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe=observation)[burn_in:]
#         add_lightweight_metropolis_hastings_duration(time.time() - start)
#         posterior_mean = posterior.mean
#
#         l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
#         add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertLess(l2_distance, 3)
#         self.assertLess(kl_divergence, 1)
#
#     def test_inference_hmm_posterior_random_walk_metropolis_hastings(self):
#         samples = lightweight_metropolis_hastings_samples
#         burn_in = lightweight_metropolis_hastings_burn_in
#         observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
#         posterior_mean_correct = self._posterior_mean_correct
#
#         start = time.time()
#         posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe=observation)[burn_in:]
#         add_random_walk_metropolis_hastings_duration(time.time() - start)
#         posterior_mean = posterior.mean
#
#         l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))
#
#         util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
#         add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertLess(l2_distance, 3)
#         self.assertLess(kl_divergence, 1)
#
#
# class BranchingTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         class Branching(Model):
#             def __init__(self):
#                 super().__init__('Branching')
#
#             @functools.lru_cache(maxsize=None)  # 128 by default
#             def fibonacci(self, n):
#                 if n < 2:
#                     return 1
#
#                 a = 1
#                 fib = 1
#                 for i in range(n-2):
#                     a, fib = fib, a + fib
#                 return fib
#
#             def forward(self):
#                 count_prior = Poisson(4)
#                 r = pyprob.sample(count_prior)
#                 if 4 < float(r):
#                     l = 6
#                 else:
#                     l = 1 + self.fibonacci(3 * int(r)) + pyprob.sample(count_prior)
#
#                 pyprob.observe(Poisson(l), name='obs')
#                 return r
#
#             def true_posterior(self, observe=6):
#                 count_prior = Poisson(4)
#                 vals = []
#                 log_weights = []
#                 for r in range(40):
#                     for s in range(40):
#                         if 4 < float(r):
#                             l = 6
#                         else:
#                             f = self.fibonacci(3 * r)
#                             l = 1 + f + count_prior.sample()
#                         vals.append(r)
#                         log_weights.append(Poisson(l).log_prob(observe) + count_prior.log_prob(r) + count_prior.log_prob(s))
#                 return Empirical(vals, log_weights)
#
#         self._model = Branching()
#         super().__init__(*args, **kwargs)
#
#     def test_inference_branching_importance_sampling(self):
#         samples = importance_sampling_samples
#         posterior_correct = util.empirical_to_categorical(self._model.true_posterior(), max_val=40)
#
#         start = time.time()
#         posterior = util.empirical_to_categorical(self._model.posterior_results(samples, observe={'obs': 6}), max_val=40)
#         add_importance_sampling_duration(time.time() - start)
#
#         posterior_probs = util.to_numpy(posterior._probs)
#         posterior_probs_correct = util.to_numpy(posterior_correct._probs)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))
#
#         util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
#         add_importance_sampling_kl_divergence(kl_divergence)
#
#         self.assertLess(kl_divergence, 0.75)
#     #
#     # def test_inference_branching_importance_sampling_with_inference_network(self):
#     #     samples = importance_sampling_samples
#     #     posterior_correct = util.empirical_to_categorical(self._model.true_posterior(), max_val=40)
#     #
#     #     self._model.reset_inference_network()
#     #     self._model.learn_inference_network(num_traces=2000, observe_embeddings={'obs': {'depth': 2, 'dim': 32}})
#     #
#     #     start = time.time()
#     #     posterior = util.empirical_to_categorical(self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs': 6}), max_val=40)
#     #     add_importance_sampling_with_inference_network_ff_duration(time.time() - start)
#     #
#     #     posterior_probs = util.to_numpy(posterior._probs)
#     #     posterior_probs_correct = util.to_numpy(posterior_correct._probs)
#     #     kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))
#     #
#     #     util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
#     #     add_importance_sampling_with_inference_network_ff_kl_divergence(kl_divergence)
#     #
#     #     self.assertLess(kl_divergence, 0.75)
#
#     def test_inference_branching_lightweight_metropolis_hastings(self):
#         samples = importance_sampling_samples
#         posterior_correct = util.empirical_to_categorical(self._model.true_posterior(), max_val=40)
#
#         start = time.time()
#         posterior = util.empirical_to_categorical(self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs': 6}), max_val=40)
#         add_lightweight_metropolis_hastings_duration(time.time() - start)
#
#         posterior_probs = util.to_numpy(posterior._probs)
#         posterior_probs_correct = util.to_numpy(posterior_correct._probs)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))
#
#         util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
#         add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertLess(kl_divergence, 0.75)
#
#     def test_inference_branching_random_walk_metropolis_hastings(self):
#         samples = importance_sampling_samples
#         posterior_correct = util.empirical_to_categorical(self._model.true_posterior(), max_val=40)
#
#         start = time.time()
#         posterior = util.empirical_to_categorical(self._model.posterior_results(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs': 6}), max_val=40)
#         add_random_walk_metropolis_hastings_duration(time.time() - start)
#
#         posterior_probs = util.to_numpy(posterior._probs)
#         posterior_probs_correct = util.to_numpy(posterior_correct._probs)
#         kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))
#
#         util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
#         add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertLess(kl_divergence, 0.75)
#
#
# class MiniCaptchaTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         class MiniCaptcha(Model):
#             def __init__(self, alphabet=['A', 'B', 'C', 'D', 'E', 'F'], noise=0.1):
#                 self._alphabet = alphabet
#                 self._probs = [1/len(alphabet) for i in range(len(alphabet))]
#                 self._noise = noise
#                 super().__init__('MiniCaptcha')
#
#             def render(self, text, size=18, height=28, width=28, x=6, y=6):
#                 pil_font = ImageFont.truetype('Ubuntu-B.ttf', size=size)
#                 text_width, text_height = pil_font.getsize(text)
#                 canvas = Image.new('RGB', [height, width], (255, 255, 255))
#                 draw = ImageDraw.Draw(canvas)
#                 draw.text((x, y), text, font=pil_font, fill='#000000')
#                 return torch.from_numpy(1 - (np.asarray(canvas) / 255.0))[:, :, 0].unsqueeze(0).float()
#
#             def forward(self):
#                 letter_id = int(pyprob.sample(Categorical(self._probs)))
#                 image = self.render(self._alphabet[letter_id]).view(-1)
#                 likelihood = Normal(image, self._noise)
#                 pyprob.observe(likelihood, name='query_image')
#                 return letter_id
#
#         self._model = MiniCaptcha()
#         self._test_images = [self._model.render(letter).view(-1) for letter in self._model._alphabet]
#         self._true_posteriors = [Categorical(util.one_hot(len(self._model._alphabet), i) + util._epsilon) for i in range(len(self._model._alphabet))]
#         super().__init__(*args, **kwargs)
#
#     def test_inference_mini_captcha_posterior_importance_sampling(self):
#         samples = int(importance_sampling_samples / len(self._model._alphabet))
#         test_letters = self._model._alphabet
#         mean_effective_sample_size_min = 0.1 * samples
#
#         start = time.time()
#         posteriors = []
#         map_estimates = []
#         effective_sample_sizes = []
#         for i in range(len(self._model._alphabet)):
#             posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'query_image': self._test_images[i]})
#             posteriors.append(posterior)
#             map_estimates.append(self._model._alphabet[int(posterior.mode)])
#             effective_sample_sizes.append(float(posterior.effective_sample_size))
#         add_importance_sampling_duration(time.time() - start)
#         mean_effective_sample_size = sum(effective_sample_sizes) / len(self._model._alphabet)
#
#         accuracy = sum([1 if map_estimates[i] == test_letters[i] else 0 for i in range(len(test_letters))])/len(test_letters)
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(util.empirical_to_categorical(p, max_val=len(self._model._alphabet)-1), tp) for (p, tp) in zip(posteriors, self._true_posteriors)]))
#
#         util.eval_print('samples', 'test_letters', 'map_estimates', 'effective_sample_sizes', 'accuracy', 'mean_effective_sample_size', 'mean_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_kl_divergence(kl_divergence)
#
#         self.assertGreater(accuracy, 0.9)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_mini_captcha_posterior_importance_sampling_with_inference_network_ff(self):
#         samples = int(importance_sampling_with_inference_network_ff_samples / len(self._model._alphabet))
#         test_letters = self._model._alphabet
#         mean_effective_sample_size_min = 0.9 * samples
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_ff_training_traces, observe_embeddings={'query_image': {'dim': 32, 'reshape': [1, 28, 28], 'embedding': ObserveEmbedding.CNN2D5C}}, prior_inflation=importance_sampling_with_inference_network_ff_prior_inflation, inference_network=InferenceNetwork.FEEDFORWARD)
#
#         # pyprob.diagnostics.network_statistics(self._model._inference_network, './report_ff')
#         start = time.time()
#         posteriors = []
#         map_estimates = []
#         effective_sample_sizes = []
#         for i in range(len(self._model._alphabet)):
#             posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'query_image': self._test_images[i]})
#             posteriors.append(posterior)
#             map_estimates.append(self._model._alphabet[int(posterior.mode)])
#             effective_sample_sizes.append(float(posterior.effective_sample_size))
#         add_importance_sampling_with_inference_network_ff_duration(time.time() - start)
#         mean_effective_sample_size = sum(effective_sample_sizes) / len(self._model._alphabet)
#
#         accuracy = sum([1 if map_estimates[i] == test_letters[i] else 0 for i in range(len(test_letters))])/len(test_letters)
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(util.empirical_to_categorical(p, max_val=len(self._model._alphabet)-1), tp) for (p, tp) in zip(posteriors, self._true_posteriors)]))
#
#         util.eval_print('samples', 'test_letters', 'map_estimates', 'effective_sample_sizes', 'accuracy', 'mean_effective_sample_size', 'mean_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_ff_kl_divergence(kl_divergence)
#
#         self.assertGreater(accuracy, 0.9)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_mini_captcha_posterior_importance_sampling_with_inference_network_lstm(self):
#         samples = int(importance_sampling_with_inference_network_lstm_samples / len(self._model._alphabet))
#         test_letters = self._model._alphabet
#         mean_effective_sample_size_min = 0.9 * samples
#
#         self._model.reset_inference_network()
#         self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_lstm_training_traces, observe_embeddings={'query_image': {'dim': 32, 'reshape': [1, 28, 28], 'embedding': ObserveEmbedding.CNN2D5C}}, prior_inflation=importance_sampling_with_inference_network_lstm_prior_inflation, inference_network=InferenceNetwork.LSTM)
#
#         # pyprob.diagnostics.network_statistics(self._model._inference_network, './report_lstm')
#         start = time.time()
#         posteriors = []
#         map_estimates = []
#         effective_sample_sizes = []
#         for i in range(len(self._model._alphabet)):
#             posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'query_image': self._test_images[i]})
#             posteriors.append(posterior)
#             map_estimates.append(self._model._alphabet[int(posterior.mode)])
#             effective_sample_sizes.append(float(posterior.effective_sample_size))
#         add_importance_sampling_with_inference_network_lstm_duration(time.time() - start)
#         mean_effective_sample_size = sum(effective_sample_sizes) / len(self._model._alphabet)
#
#         accuracy = sum([1 if map_estimates[i] == test_letters[i] else 0 for i in range(len(test_letters))])/len(test_letters)
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(util.empirical_to_categorical(p, max_val=len(self._model._alphabet)-1), tp) for (p, tp) in zip(posteriors, self._true_posteriors)]))
#
#         util.eval_print('samples', 'test_letters', 'map_estimates', 'effective_sample_sizes', 'accuracy', 'mean_effective_sample_size', 'mean_effective_sample_size_min', 'kl_divergence')
#         add_importance_sampling_with_inference_network_lstm_kl_divergence(kl_divergence)
#
#         self.assertGreater(accuracy, 0.9)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_mini_captcha_posterior_lightweight_metropolis_hastings(self):
#         samples = int(lightweight_metropolis_hastings_samples / len(self._model._alphabet))
#         burn_in = int(lightweight_metropolis_hastings_burn_in / len(self._model._alphabet))
#         test_letters = self._model._alphabet
#
#         start = time.time()
#         posteriors = []
#         map_estimates = []
#         for i in range(len(self._model._alphabet)):
#             posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'query_image': self._test_images[i]})[burn_in:]
#             posteriors.append(posterior)
#             map_estimates.append(self._model._alphabet[int(posterior.combine_duplicates().mode)])
#         add_lightweight_metropolis_hastings_duration(time.time() - start)
#
#         accuracy = sum([1 if map_estimates[i] == test_letters[i] else 0 for i in range(len(test_letters))])/len(test_letters)
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(util.empirical_to_categorical(p, max_val=len(self._model._alphabet)-1), tp) for (p, tp) in zip(posteriors, self._true_posteriors)]))
#
#         util.eval_print('samples', 'test_letters', 'map_estimates', 'accuracy', 'kl_divergence')
#         add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertGreater(accuracy, 0.9)
#         self.assertLess(kl_divergence, 0.25)
#
#     def test_inference_mini_captcha_posterior_random_walk_metropolis_hastings(self):
#         samples = int(random_walk_metropolis_hastings_samples / len(self._model._alphabet))
#         burn_in = int(random_walk_metropolis_hastings_burn_in / len(self._model._alphabet))
#         test_letters = self._model._alphabet
#
#         start = time.time()
#         posteriors = []
#         map_estimates = []
#         for i in range(len(self._model._alphabet)):
#             posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'query_image': self._test_images[i]})[burn_in:]
#             posteriors.append(posterior)
#             map_estimates.append(self._model._alphabet[int(posterior.combine_duplicates().mode)])
#         add_random_walk_metropolis_hastings_duration(time.time() - start)
#
#         accuracy = sum([1 if map_estimates[i] == test_letters[i] else 0 for i in range(len(test_letters))])/len(test_letters)
#         kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(util.empirical_to_categorical(p, max_val=len(self._model._alphabet)-1), tp) for (p, tp) in zip(posteriors, self._true_posteriors)]))
#
#         util.eval_print('samples', 'test_letters', 'map_estimates', 'accuracy', 'kl_divergence')
#         add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)
#
#         self.assertGreater(accuracy, 0.9)
#         self.assertLess(kl_divergence, 0.25)
#
#
# if __name__ == '__main__':
#     pyprob.seed(123)
#     pyprob.set_verbosity(2)
#
#     tests = []
#     tests.append('GaussianWithUnknownMeanTestCase')
#     tests.append('GaussianWithUnknownMeanMarsagliaTestCase')
#     # tests.append('HiddenMarkovModelTestCase')
#     # tests.append('BranchingTestCase')
#     tests.append('MiniCaptchaTestCase')
#
#     time_start = time.time()
#     success = unittest.main(defaultTest=tests, verbosity=2, exit=False).result.wasSuccessful()
#     print('\nDuration                   : {}'.format(util.days_hours_mins_secs_str(time.time() - time_start)))
#     print('Models run                 : {}'.format(' '.join(tests)))
#     print('\nTotal inference performance:\n')
#     print(colored('                                              Samples        KL divergence  Duration (s) ', 'yellow', attrs=['bold']))
#     print(colored('Importance sampling                         : ', 'yellow', attrs=['bold']), end='')
#     print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_samples, importance_sampling_kl_divergence, importance_sampling_duration), 'white', attrs=['bold']))
#     print(colored('Importance sampling w/ inference net. (FF)  : ', 'yellow', attrs=['bold']), end='')
#     print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_with_inference_network_ff_samples, importance_sampling_with_inference_network_ff_kl_divergence, importance_sampling_with_inference_network_ff_duration), 'white', attrs=['bold']))
#     print(colored('Importance sampling w/ inference net. (LSTM): ', 'yellow', attrs=['bold']), end='')
#     print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_with_inference_network_lstm_samples, importance_sampling_with_inference_network_lstm_kl_divergence, importance_sampling_with_inference_network_lstm_duration), 'white', attrs=['bold']))
#     print(colored('Lightweight Metropolis Hastings             : ', 'yellow', attrs=['bold']), end='')
#     print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(lightweight_metropolis_hastings_samples, lightweight_metropolis_hastings_kl_divergence, lightweight_metropolis_hastings_duration), 'white', attrs=['bold']))
#     print(colored('Random-walk Metropolis Hastings             : ', 'yellow', attrs=['bold']), end='')
#     print(colored('{:+.6e}  {:+.6e}  {:+.6e}\n'.format(random_walk_metropolis_hastings_samples, random_walk_metropolis_hastings_kl_divergence, random_walk_metropolis_hastings_duration), 'white', attrs=['bold']))
#     sys.exit(0 if success else 1)
