import torch.nn.functional as F
import unittest
import math
import time
import sys
import docker
import functools
from termcolor import colored

import pyprob
from pyprob import util, RemoteModel, InferenceEngine
from pyprob.distributions import Normal, Categorical, Poisson, Empirical


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


docker_client = docker.from_env()
print('Pulling latest Docker image: probprog/pyprob_cpp')
docker_client.images.pull('probprog/pyprob_cpp')
print('Docker image pulled.')

docker_containers = []
docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_gum ipc://@GaussianWithUnknownMeanCPP', network='host', detach=True))
GaussianWithUnknownMeanCPP = RemoteModel('ipc://@GaussianWithUnknownMeanCPP')

docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_gum_marsaglia_replacement ipc://@GaussianWithUnknownMeanMarsagliaWithReplacementCPP', network='host', detach=True))
GaussianWithUnknownMeanMarsagliaWithReplacementCPP = RemoteModel('ipc://@GaussianWithUnknownMeanMarsagliaWithReplacementCPP')

docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_hmm ipc://@HiddenMarkovModelCPP', network='host', detach=True))
HiddenMarkovModelCPP = RemoteModel('ipc://@HiddenMarkovModelCPP')

docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_branching ipc://@BranchingCPP', network='host', detach=True))
BranchingCPP = RemoteModel('ipc://@BranchingCPP')


class GaussianWithUnknownMeanTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = GaussianWithUnknownMeanCPP
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
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
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
        posterior_effective_sample_size_min = samples * 0.03

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs0': {'dim': 256, 'depth': 1}, 'obs1': {'dim': 256, 'depth': 1}})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
        add_importance_sampling_with_inference_network_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
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
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
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
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class GaussianWithUnknownMeanMarsagliaWithReplacementTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = GaussianWithUnknownMeanMarsagliaWithReplacementCPP
        super().__init__(*args, **kwargs)

    def test_inference_gum_marsaglia_replacement_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.005

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_replacement_posterior_importance_sampling_with_inference_network(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        posterior_effective_sample_size_min = samples * 0.03

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs0': {'dim': 256, 'depth': 1}, 'obs1': {'dim': 256, 'depth': 1}})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
        add_importance_sampling_with_inference_network_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
        add_importance_sampling_with_inference_network_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_replacement_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_inference_gum_marsaglia_replacement_posterior_random_walk_metropolis_hastings(self):
        samples = random_walk_metropolis_hastings_samples
        burn_in = random_walk_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class HiddenMarkovModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = HiddenMarkovModelCPP
        self._observation = [0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]
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

        print(posterior[0])
        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)

    def test_inference_hmm_posterior_importance_sampling_with_inference_network(self):
        samples = importance_sampling_with_inference_network_samples
        observation = {'obs{}'.format(i): self._observation[i] for i in range(len(self._observation))}
        posterior_mean_correct = self._posterior_mean_correct
        posterior_effective_sample_size_min = samples * 0.03

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=importance_sampling_with_inference_network_training_traces, observe_embeddings={'obs{}'.format(i): {'depth': 2, 'dim': 16} for i in range(len(observation))})

        start = time.time()
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe=observation)
        add_importance_sampling_with_inference_network_duration(time.time() - start)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean
        posterior_effective_sample_size = float(posterior.effective_sample_size)

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())
        kl_divergence = float(sum([pyprob.distributions.Distribution.kl_divergence(Categorical(i + util._epsilon), Categorical(j + util._epsilon)) for (i, j) in zip(posterior_mean, posterior_mean_correct)]))

        util.eval_print('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'l2_distance', 'kl_divergence')
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

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
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

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'l2_distance', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(l2_distance, 3)
        self.assertLess(kl_divergence, 1)


class BranchingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = BranchingCPP
        super().__init__(*args, **kwargs)

    @functools.lru_cache(maxsize=None)  # 128 by default
    def fibonacci(self, n):
        if n < 2:
            return 1

        a = 1
        fib = 1
        for i in range(n-2):
            a, fib = fib, a + fib
        return fib

    def true_posterior(self, observe=6):
        count_prior = Poisson(4)
        vals = []
        log_weights = []
        for r in range(40):
            for s in range(40):
                if 4 < float(r):
                    l = 6
                else:
                    f = self.fibonacci(3 * r)
                    l = 1 + f + count_prior.sample()
                vals.append(r)
                log_weights.append(Poisson(l).log_prob(observe) + count_prior.log_prob(r) + count_prior.log_prob(s))
        return Empirical(vals, log_weights)

    def test_inference_branching_importance_sampling(self):
        samples = importance_sampling_samples
        posterior_correct = util.empirical_to_categorical(self.true_posterior(), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, observe={'obs': 6}), max_val=40)
        add_importance_sampling_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs)
        posterior_probs_correct = util.to_numpy(posterior_correct._probs)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))

        util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)
    #
    # def test_inference_branching_importance_sampling_with_inference_network(self):
    #     samples = importance_sampling_samples
    #     posterior_correct = util.empirical_to_categorical(self._model.true_posterior(), max_val=40)
    #
    #     self._model.reset_inference_network()
    #     self._model.learn_inference_network(num_traces=2000, observe_embeddings={'obs': {'depth': 2, 'dim': 32}})
    #
    #     start = time.time()
    #     posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs': 6}), max_val=40)
    #     add_importance_sampling_with_inference_network_duration(time.time() - start)
    #
    #     posterior_probs = util.to_numpy(posterior._probs)
    #     posterior_probs_correct = util.to_numpy(posterior_correct._probs)
    #     kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))
    #
    #     util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
    #     add_importance_sampling_with_inference_network_kl_divergence(kl_divergence)
    #
    #     self.assertLess(kl_divergence, 0.75)

    def test_inference_branching_lightweight_metropolis_hastings(self):
        samples = importance_sampling_samples
        posterior_correct = util.empirical_to_categorical(self.true_posterior(), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs': 6}), max_val=40)
        add_lightweight_metropolis_hastings_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs)
        posterior_probs_correct = util.to_numpy(posterior_correct._probs)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))

        util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_lightweight_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)

    def test_inference_branching_random_walk_metropolis_hastings(self):
        samples = importance_sampling_samples
        posterior_correct = util.empirical_to_categorical(self.true_posterior(), max_val=40)

        start = time.time()
        posterior = util.empirical_to_categorical(self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs': 6}), max_val=40)
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_probs = util.to_numpy(posterior._probs)
        posterior_probs_correct = util.to_numpy(posterior_correct._probs)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(posterior, posterior_correct))

        util.eval_print('samples', 'posterior_probs', 'posterior_probs_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertLess(kl_divergence, 0.75)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    tests = []
    tests.append('GaussianWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanMarsagliaWithReplacementTestCase')
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

    # for container in docker_containers:
    #     print('Killing Docker container {}'.format(container.name))
    #     container.kill()

    sys.exit(0 if success else 1)
