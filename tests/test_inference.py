import unittest
import sys
import math
import torch
import torch.nn.functional as F
from termcolor import colored
import time
import numpy as np
import functools

import pyprob
from pyprob import util
from pyprob import Model
from pyprob.distributions import Categorical, Normal, Uniform, Empirical, Poisson


importance_sampling_samples = 5000
importance_sampling_kl_divergence = 0
importance_sampling_duration = 0

inference_compilation_samples = 5000
inference_compilation_kl_divergence = 0
inference_compilation_duration = 0
inference_compilation_training_traces = 20000

lightweight_metropolis_hastings_samples = 5000
lightweight_metropolis_hastings_kl_divergence = 0
lightweight_metropolis_hastings_duration = 0

random_walk_metropolis_hastings_samples = 5000
random_walk_metropolis_hastings_kl_divergence = 0
random_walk_metropolis_hastings_duration = 0


def add_importance_sampling_kl_divergence(val):
    global importance_sampling_kl_divergence
    importance_sampling_kl_divergence += val


def add_inference_compilation_kl_divergence(val):
    global inference_compilation_kl_divergence
    inference_compilation_kl_divergence += val


def add_lightweight_metropolis_hastings_kl_divergence(val):
    global lightweight_metropolis_hastings_kl_divergence
    lightweight_metropolis_hastings_kl_divergence += val


def add_random_walk_metropolis_hastings_kl_divergence(val):
    global random_walk_metropolis_hastings_kl_divergence
    random_walk_metropolis_hastings_kl_divergence += val


def add_importance_sampling_duration(val):
    global importance_sampling_duration
    importance_sampling_duration += val


def add_inference_compilation_duration(val):
    global inference_compilation_duration
    inference_compilation_duration += val


def add_lightweight_metropolis_hastings_duration(val):
    global lightweight_metropolis_hastings_duration
    lightweight_metropolis_hastings_duration += val


def add_random_walk_metropolis_hastings_duration(val):
    global random_walk_metropolis_hastings_duration
    random_walk_metropolis_hastings_duration += val


# class MVNWithUnknownMeanTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         class MVNWithUnknownMean(Model):
#             def __init__(self, prior_mean=[[1, 1]], prior_stddev=[[math.sqrt(5), math.sqrt(5)]], likelihood_stddev=[[math.sqrt(2), math.sqrt(2)]]):
#                 self.prior_mean = prior_mean
#                 self.prior_stddev = prior_stddev
#                 self.likelihood_stddev = likelihood_stddev
#                 super().__init__('Gaussian with unknown mean')
#
#             def forward(self, observation=[]):
#                 mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
#                 likelihood = Normal(mu, self.likelihood_stddev)
#                 for o in observation:
#                     pyprob.observe(likelihood, o)
#                 return mu
#
#         self._model = MVNWithUnknownMean()
#         super().__init__(*args, **kwargs)
#
#     def test_inference_mvnum_posterior_importance_sampling(self):
#         samples = 10000
#
#         observation = [[8, 8], [9, 9]]
#         posterior_mean_correct = [[7.25, 7.25]]
#         posterior_stddev_correct = [[math.sqrt(1/1.2), math.sqrt(1/1.2)]]
#
#         posterior = self._model.posterior_distribution(samples, observation=observation)
#         posterior_mean = util.to_numpy(posterior.mean)
#         posterior_mean_unweighted = util.to_numpy(posterior.unweighted().mean)
#         posterior_stddev = util.to_numpy(posterior.stddev)
#         posterior_stddev_unweighted = util.to_numpy(posterior.unweighted().stddev)
#         # kl_divergence = util.to_numpy(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))
#         # add_importance_sampling_kl_divergence(kl_divergence)
#
#         util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct')
#         self.assertTrue(np.allclose(posterior_mean, posterior_mean_correct, atol=0.1))
#         self.assertTrue(np.allclose(posterior_stddev, posterior_stddev_correct, atol=0.1))
#         # self.assertLess(kl_divergence, 0.15)
#
#     def test_inference_mvnum_posterior_metropolis_hastings(self):
#         samples = 1000000
#
#         observation = [[8, 8], [9, 9]]
#         posterior_mean_correct = [[7.25, 7.25]]
#         posterior_stddev_correct = [[math.sqrt(1/1.2), math.sqrt(1/1.2)]]
#
#         posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observation=observation)
#         posterior_mean = util.to_numpy(posterior.mean)
#         posterior_mean_unweighted = util.to_numpy(posterior.unweighted().mean)
#         posterior_stddev = util.to_numpy(posterior.stddev)
#         posterior_stddev_unweighted = util.to_numpy(posterior.unweighted().stddev)
#         # kl_divergence = util.to_numpy(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))
#         # add_importance_sampling_kl_divergence(kl_divergence)
#
#         util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct')
#         self.assertTrue(np.allclose(posterior_mean, posterior_mean_correct, atol=0.1))
#         self.assertTrue(np.allclose(posterior_stddev, posterior_stddev_correct, atol=0.1))
#         # self.assertLess(kl_divergence, 0.15)

    # def test_inference_mvnum_posterior_inference_compilation(self):
    #     observation = [[8, 8], [9, 9]]
    #     posterior_mean_correct = [[7.25, 7.25]]
    #     posterior_stddev_correct = [[math.sqrt(1/1.2), math.sqrt(1/1.2)]]
    #
    #     self._model.learn_inference_network(observation=[[1,1],[1,1]], num_traces=training_traces)
    #     posterior = self._model.posterior_distribution(samples, observation=observation)
    #     posterior_mean = util.to_numpy(posterior.mean)
    #     posterior_mean_unweighted = util.to_numpy(posterior.unweighted().mean)
    #     posterior_stddev = util.to_numpy(posterior.stddev)
    #     posterior_stddev_unweighted = util.to_numpy(posterior.unweighted().stddev)
    #     # kl_divergence = util.to_numpy(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))
    #     # add_importance_sampling_kl_divergence(kl_divergence)
    #
    #     util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct')
    #     self.assertTrue(np.allclose(posterior_mean, posterior_mean_correct, atol=0.1))
    #     self.assertTrue(np.allclose(posterior_stddev, posterior_stddev_correct, atol=0.1))
    #     # self.assertLess(kl_divergence, 0.15)


class GaussianWithUnknownMeanTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self, observation=[]):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_inference_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, observation=observation)
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_inference_compilation(self):
        samples = inference_compilation_samples
        training_traces = inference_compilation_training_traces
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1, 1], num_traces=training_traces, prior_inflation=pyprob.PriorInflation.ENABLED)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observation=observation)
        add_inference_compilation_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_inference_compilation_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_lightweight_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observation=observation)
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observation=observation)
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class GaussianWithUnknownMeanMarsagliaTestCase(unittest.TestCase):
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
                    x = pyprob.sample(uniform, replace=True)[0]
                    y = pyprob.sample(uniform, replace=True)[0]
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

    def test_inference_gum_marsaglia_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, observation=observation)
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_inference_compilation(self):
        samples = inference_compilation_samples
        training_traces = inference_compilation_training_traces
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1, 1], num_traces=training_traces, prior_inflation=pyprob.PriorInflation.ENABLED)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observation=observation)
        add_inference_compilation_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_inference_compilation_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observation=observation)
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        observation = [8, 9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observation=observation)
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(Normal(posterior_mean_correct, posterior_stddev_correct), Normal(posterior.mean, posterior_stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class HiddenMarkovModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class HiddenMarkovModel(Model):
            def __init__(self, init_dist, trans_dists, obs_dists):
                self.init_dist = init_dist
                self.trans_dists = trans_dists
                self.obs_dists = obs_dists
                super().__init__('Hidden Markov model')

            def forward(self, observation=[]):
                states = [pyprob.sample(init_dist)]
                for o in observation:
                    state = pyprob.sample(self.trans_dists[int(states[-1])])
                    pyprob.observe(self.obs_dists[int(state)], o)
                    states.append(state)
                return torch.stack([util.one_hot(3, int(s)) for s in states])

        init_dist = Categorical([1, 1, 1])
        trans_dists = [Categorical([0.1, 0.5, 0.4]),
                       Categorical([0.2, 0.2, 0.6]),
                       Categorical([0.15, 0.15, 0.7])]
        obs_dists = [Normal(-1, 1),
                     Normal(1, 1),
                     Normal(0, 1)]
        self._model = HiddenMarkovModel(init_dist, trans_dists, obs_dists)

        self._observation = [0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]
        self._posterior_mean_correct = util.to_variable([[0.3775, 0.3092, 0.3133],
                                                         [0.0416, 0.4045, 0.5539],
                                                         [0.0541, 0.2552, 0.6907],
                                                         [0.0455, 0.2301, 0.7244],
                                                         [0.1062, 0.1217, 0.7721],
                                                         [0.0714, 0.1732, 0.7554],
                                                         [0.9300, 0.0001, 0.0699],
                                                         [0.4577, 0.0452, 0.4971],
                                                         [0.0926, 0.2169, 0.6905],
                                                         [0.1014, 0.1359, 0.7626],
                                                         [0.0985, 0.1575, 0.7440],
                                                         [0.1781, 0.2198, 0.6022],
                                                         [0.0000, 0.9848, 0.0152],
                                                         [0.1130, 0.1674, 0.7195],
                                                         [0.0557, 0.1848, 0.7595],
                                                         [0.2017, 0.0472, 0.7511],
                                                         [0.2545, 0.0611, 0.6844]])
        super().__init__(*args, **kwargs)

    def test_inference_hmm_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        start = time.time()
        posterior = self._model.posterior_distribution(samples, observation=observation)
        add_importance_sampling_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([util.kl_divergence_categorical(Categorical(i), Categorical(j)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_inference_compilation(self):
        samples = inference_compilation_samples
        training_traces = inference_compilation_training_traces
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        self._model.learn_inference_network(observation=torch.zeros(16), num_traces=training_traces, prior_inflation=pyprob.PriorInflation.DISABLED)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observation=observation)
        add_inference_compilation_duration(time.time() - start)

        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([util.kl_divergence_categorical(Categorical(i), Categorical(j)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_inference_compilation_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observation=observation)
        add_lightweight_metropolis_hastings_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([util.kl_divergence_categorical(Categorical(i), Categorical(j)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observation=observation)
        add_random_walk_metropolis_hastings_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([util.kl_divergence_categorical(Categorical(i), Categorical(j)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)


class BranchingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class Branching(Model):
            def __init__(self):
                super().__init__('Branching')

            @functools.lru_cache(maxsize=None)  # 128 by default
            def fibonacci(self, n):
                if n < 2:
                    return 1

                a = 1
                fib = 1
                for i in range(n-2):
                    a, fib = fib, a + fib
                return fib

            def forward(self, observation=0):
                count_prior = Poisson(4)
                r = pyprob.sample(count_prior)
                if 4 < float(r):
                    l = 6
                else:
                    l = 1 + self.fibonacci(3 * int(r)) + pyprob.sample(count_prior)

                pyprob.observe(Poisson(l), observation)
                return r

            def true_posterior(self, observation=6):
                count_prior = Poisson(4)
                vals = []
                log_weights = []
                for r in range(40):
                    for s in range(40):
                        if 4 < float(r):
                            l = 6
                        else:
                            f = self.fibonacci(3 * r)
                            l = 1 + f + pyprob.sample(count_prior)
                        vals.append(r)
                        log_weights.append(Poisson(l).log_prob(observation) + count_prior.log_prob(r) + count_prior.log_prob(s))
                return Empirical(vals, log_weights)

        self._model = Branching()
        super().__init__(*args, **kwargs)

    def test_inference_branching_importance_sampling(self):
        samples = importance_sampling_samples
        observation = 6
        posterior_correct = util.empirical_to_categorical(self._model.true_posterior(observation), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, observation=observation), max_val=40)
        add_importance_sampling_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs[0])
        posterior_probs_correct = util.to_numpy(posterior_correct._probs[0])

        kl_divergence = float(util.kl_divergence_categorical(posterior_correct, posterior))

        util.debug('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)

    def test_inference_branching_inference_compilation(self):
        samples = inference_compilation_samples
        observation = 6
        training_traces = inference_compilation_training_traces
        posterior_correct = util.empirical_to_categorical(self._model.true_posterior(observation), max_val=40)

        self._model.learn_inference_network(observation=1, num_traces=training_traces)
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, observation=observation, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK), max_val=40)
        posterior_probs = util.to_numpy(posterior._probs[0])
        posterior_probs_correct = util.to_numpy(posterior_correct._probs[0])

        kl_divergence = float(util.kl_divergence_categorical(posterior_correct, posterior))

        util.debug('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)

    def test_inference_branching_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        observation = 6
        posterior_correct = util.empirical_to_categorical(self._model.true_posterior(observation), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, observation=observation, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS), max_val=40)
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs[0])
        posterior_probs_correct = util.to_numpy(posterior_correct._probs[0])

        kl_divergence = float(util.kl_divergence_categorical(posterior_correct, posterior))

        util.debug('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)

    def test_inference_branching_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        observation = 6
        posterior_correct = util.empirical_to_categorical(self._model.true_posterior(observation), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, observation=observation, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS), max_val=40)
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs[0])
        posterior_probs_correct = util.to_numpy(posterior_correct._probs[0])

        kl_divergence = float(util.kl_divergence_categorical(posterior_correct, posterior))

        util.debug('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)


if __name__ == '__main__':
    # if torch.cuda.is_available():
        # pyprob.set_cuda(True)
    pyprob.set_verbosity(2)
    tests = []
    # tests.append('MVNWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanMarsagliaTestCase')
    tests.append('HiddenMarkovModelTestCase')
    tests.append('BranchingTestCase')

    time_start = time.time()
    success = unittest.main(defaultTest=tests, verbosity=2, exit=False).result.wasSuccessful()
    print('\nDuration                   : {}'.format(util.days_hours_mins_secs_str(time.time() - time_start)))
    print('Models run                 : {}'.format(' '.join(tests)))
    print('\nTotal inference performance:\n')
    print(colored('                                 Samples        KL divergence  Duration (s) ', 'yellow', attrs=['bold']))
    print(colored('Importance sampling            : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_samples, importance_sampling_kl_divergence, importance_sampling_duration), 'white', attrs=['bold']))
    print(colored('Inference compilation          : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(inference_compilation_samples, inference_compilation_kl_divergence, inference_compilation_duration), 'white', attrs=['bold']))
    print(colored('Lightweight Metropolis Hastings: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(lightweight_metropolis_hastings_samples, lightweight_metropolis_hastings_kl_divergence, lightweight_metropolis_hastings_duration), 'white', attrs=['bold']))
    print(colored('Random-walk Metropolis Hastings: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}\n'.format(random_walk_metropolis_hastings_samples, random_walk_metropolis_hastings_kl_divergence, random_walk_metropolis_hastings_duration), 'white', attrs=['bold']))
    sys.exit(0 if success else 1)
