import unittest
import torch
import torch.nn.functional as F
import time
import math
import sys
from termcolor import colored

import pyprob
from pyprob import util, Model, InferenceEngine
from pyprob.distributions import Normal, Uniform, Categorical


importance_sampling_samples = 5000
importance_sampling_kl_divergence = 0
importance_sampling_duration = 0

importance_sampling_with_inference_network_samples = 5000
importance_sampling_with_inference_network_kl_divergence = 0
importance_sampling_with_inference_network_duration = 0
importance_sampling_with_inference_network_training_traces = 25000

lightweight_metropolis_hastings_samples = 5000
lightweight_metropolis_hastings_burn_in = 500
lightweight_metropolis_hastings_kl_divergence = 0
lightweight_metropolis_hastings_duration = 0

random_walk_metropolis_hastings_samples = 5000
random_walk_metropolis_hastings_burn_in = 500
random_walk_metropolis_hastings_kl_divergence = 0
random_walk_metropolis_hastings_duration = 0


def add_importance_sampling_kl_divergence(val):
    global importance_sampling_kl_divergence
    importance_sampling_kl_divergence += val


def add_importance_sampling_with_inference_network_kl_divergence(val):
    global importance_sampling_with_inference_network_kl_divergence
    importance_sampling_with_inference_network_kl_divergence += val


def add_lightweight_metropolis_hastings_kl_divergence(val):
    global lightweight_metropolis_hastings_kl_divergence
    lightweight_metropolis_hastings_kl_divergence += val


def add_random_walk_metropolis_hastings_kl_divergence(val):
    global random_walk_metropolis_hastings_kl_divergence
    random_walk_metropolis_hastings_kl_divergence += val


def add_importance_sampling_duration(val):
    global importance_sampling_duration
    importance_sampling_duration += val


def add_importance_sampling_with_inference_network_duration(val):
    global importance_sampling_with_inference_network_duration
    importance_sampling_with_inference_network_duration += val


def add_lightweight_metropolis_hastings_duration(val):
    global lightweight_metropolis_hastings_duration
    lightweight_metropolis_hastings_duration += val


def add_random_walk_metropolis_hastings_duration(val):
    global random_walk_metropolis_hastings_duration
    random_walk_metropolis_hastings_duration += val


class GaussianWithUnknownMeanTestCase(unittest.TestCase):
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
                pyprob.observe(likelihood, name='obs1')
                pyprob.observe(likelihood, name='obs2')
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_inference_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.005

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs1': 8, 'obs2': 9})
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_importance_sampling_with_inference_network(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        posterior_effective_sample_size_min = samples * 0.6

        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs1': {'dim': 256, 'depth': 1}, 'obs2': {'dim': 256, 'depth': 1}})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs1': 8, 'obs2': 9})
        add_importance_sampling_with_inference_network_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_with_inference_network_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs1': 8, 'obs2': 9})[burn_in:]
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        burn_in = random_walk_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs1': 8, 'obs2': 9})[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
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
                    x = pyprob.sample(uniform, replace=True)
                    y = pyprob.sample(uniform, replace=True)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, name='obs1')
                pyprob.observe(likelihood, name='obs2')
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_inference_gum_marsaglia_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.005

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs1': 8, 'obs2': 9})
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_importance_sampling_with_inference_network(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        posterior_effective_sample_size_min = samples * 0.03

        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs1': {'dim': 256, 'depth': 1}, 'obs2': {'dim': 256, 'depth': 1}})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs1': 8, 'obs2': 9})
        add_importance_sampling_with_inference_network_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_with_inference_network_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs1': 8, 'obs2': 9})[burn_in:]
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        burn_in = random_walk_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs1': 8, 'obs2': 9})[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class HiddenMarkovModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class HiddenMarkovModel(Model):
            def __init__(self, init_dist, trans_dists, obs_dists, obs_length):
                self.init_dist = init_dist
                self.trans_dists = trans_dists
                self.obs_dists = obs_dists
                self.obs_length = obs_length
                super().__init__('Hidden Markov model')

            def forward(self):
                states = [pyprob.sample(init_dist)]
                for i in range(self.obs_length):
                    state = pyprob.sample(self.trans_dists[int(states[-1])])
                    pyprob.observe(self.obs_dists[int(state)], name='obs{}'.format(i))
                    states.append(state)
                return torch.stack([util.one_hot(3, int(s)) for s in states])

        init_dist = Categorical([1, 1, 1])
        trans_dists = [Categorical([0.1, 0.5, 0.4]),
                       Categorical([0.2, 0.2, 0.6]),
                       Categorical([0.15, 0.15, 0.7])]
        obs_dists = [Normal(-1, 1),
                     Normal(1, 1),
                     Normal(0, 1)]

        self._observation = [0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]
        self._model = HiddenMarkovModel(init_dist, trans_dists, obs_dists, len(self._observation))
        self._posterior_mean_correct = util.to_tensor([[0.3775, 0.3092, 0.3133],
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
        observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
        posterior_mean_correct = self._posterior_mean_correct
        posterior_effective_sample_size_min = samples * 0.0015

        start = time.time()
        posterior = self._model.posterior_distribution(samples, observe=observation)
        add_importance_sampling_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean
        posterior_effective_sample_size = float(posterior.effective_sample_size)

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_importance_sampling_with_inference_network(self):
        samples = importance_sampling_with_inference_network_samples
        observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
        posterior_mean_correct = self._posterior_mean_correct
        posterior_effective_sample_size_min = samples * 0.1

        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs{}'.format(i): {'depth': 2, 'dim': 16} for i in range(len(observation))})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe=observation)
        add_importance_sampling_with_inference_network_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean
        posterior_effective_sample_size = float(posterior.effective_sample_size)

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
        add_importance_sampling_with_inference_network_kl_divergence(kl_divergence)

        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
        posterior_mean_correct = self._posterior_mean_correct

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe=observation)[burn_in:]
        add_lightweight_metropolis_hastings_duration(time.time() - start)
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_random_walk_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
        posterior_mean_correct = self._posterior_mean_correct

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe=observation)[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(2)
    tests = []
    tests.append('GaussianWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanMarsagliaTestCase')
    tests.append('HiddenMarkovModelTestCase')
    # tests.append('BranchingTestCase')

    time_start = time.time()
    success = unittest.main(defaultTest=tests, verbosity=2, exit=False).result.wasSuccessful()
    print('\nDuration                   : {}'.format(util.days_hours_mins_secs_str(time.time() - time_start)))
    print('Models run                 : {}'.format(' '.join(tests)))
    print('\nTotal inference performance:\n')
    print(colored('                                       Samples        KL divergence  Duration (s) ', 'yellow', attrs=['bold']))
    print(colored('Importance sampling                  : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_samples, importance_sampling_kl_divergence, importance_sampling_duration), 'white', attrs=['bold']))
    print(colored('Importance sampling w/ inference net.: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_with_inference_network_samples, importance_sampling_with_inference_network_kl_divergence, importance_sampling_with_inference_network_duration), 'white', attrs=['bold']))
    print(colored('Lightweight Metropolis Hastings      : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(lightweight_metropolis_hastings_samples, lightweight_metropolis_hastings_kl_divergence, lightweight_metropolis_hastings_duration), 'white', attrs=['bold']))
    print(colored('Random-walk Metropolis Hastings      : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}\n'.format(random_walk_metropolis_hastings_samples, random_walk_metropolis_hastings_kl_divergence, random_walk_metropolis_hastings_duration), 'white', attrs=['bold']))
    sys.exit(0 if success else 1)
