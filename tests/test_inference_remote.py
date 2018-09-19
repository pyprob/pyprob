import unittest
import math
import time
import sys
import docker
from termcolor import colored

import pyprob
from pyprob import util, ModelRemote, InferenceEngine
from pyprob.distributions import Normal


importance_sampling_samples = 500
importance_sampling_kl_divergence = 0
importance_sampling_duration = 0

importance_sampling_with_inference_network_samples = 500
importance_sampling_with_inference_network_kl_divergence = 0
importance_sampling_with_inference_network_duration = 0
importance_sampling_with_inference_network_training_traces = 500

lightweight_metropolis_hastings_samples = 500
lightweight_metropolis_hastings_burn_in = 100
lightweight_metropolis_hastings_kl_divergence = 0
lightweight_metropolis_hastings_duration = 0

random_walk_metropolis_hastings_samples = 500
random_walk_metropolis_hastings_burn_in = 100
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
docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_gum tcp://*:5555', network='host', detach=True))
GaussianWithUnknownMeanCPP = ModelRemote('tcp://127.0.0.1:5555')

docker_containers.append(docker_client.containers.run('probprog/pyprob_cpp', '/code/pyprob_cpp/build/pyprob_cpp/test_gum_marsaglia_replacement tcp://*:5556', network='host', detach=True))
GaussianWithUnknownMeanMarsagliaWithReplacementCPP = ModelRemote('tcp://127.0.0.1:5556')


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
        posterior_effective_sample_size_min = samples * 0.03

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
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
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
        posterior = self._model.posterior_distribution(samples, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]
        add_random_walk_metropolis_hastings_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
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

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
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

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')
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

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
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

        util.debug('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_random_walk_metropolis_hastings_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(2)
    tests = []
    tests.append('GaussianWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanMarsagliaWithReplacementTestCase')
    # tests.append('HiddenMarkovModelTestCase')
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

    for container in docker_containers:
        print('Killing Docker container {}'.format(container.name))
        container.kill()
    sys.exit(0 if success else 1)
