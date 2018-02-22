import unittest
import math
import uuid
import tempfile
import os
import sys
import traceback
import docker

from pyprob import ModelRemote
from pyprob import util


docker_client = docker.from_env()
print('Pulling latest Docker image: probprog/cpproblight')
docker_client.images.pull('probprog/cpproblight')
print('Docker image pulled.')

docker_client.containers.run('probprog/cpproblight', '/code/cpproblight/build/cpproblight/test_gum_marsaglia tcp://*:5555', network='host', detach=True)
GaussianWithUnknownMeanMarsagliaCPP = ModelRemote('tcp://127.0.0.1:5555')

docker_client.containers.run('probprog/cpproblight', '/code/cpproblight/build/cpproblight/test_gum_marsaglia_replacement tcp://*:5556', network='host', detach=True)
GaussianWithUnknownMeanMarsagliaWithReplacementCPP = ModelRemote('tcp://127.0.0.1:5556')


class ModelRemoteTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = GaussianWithUnknownMeanMarsagliaCPP
        super().__init__(*args, **kwargs)

    def test_model_remote_prior(self):
        samples = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')
        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_remote_trace_length_statistics(self):
        samples = 2000
        trace_length_mean_correct = 2.5630438327789307
        trace_length_stddev_correct = 1.2081329822540283
        trace_length_min_correct = 2

        trace_length_mean = float(self._model.trace_length_mean(samples))
        trace_length_stddev = float(self._model.trace_length_stddev(samples))
        trace_length_min = float(self._model.trace_length_min(samples))
        trace_length_max = float(self._model.trace_length_max(samples))

        util.debug('samples', 'trace_length_mean', 'trace_length_mean_correct', 'trace_length_stddev', 'trace_length_stddev_correct', 'trace_length_min', 'trace_length_min_correct', 'trace_length_max')

        self.assertAlmostEqual(trace_length_mean, trace_length_mean_correct, places=0)
        self.assertAlmostEqual(trace_length_stddev, trace_length_stddev_correct, places=0)
        self.assertAlmostEqual(trace_length_min, trace_length_min_correct, places=0)

    def test_model_remote_train_save_load(self):
        training_traces = 128
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        self._model.learn_inference_network(observation=[1, 1], early_stop_traces=training_traces)
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)

        util.debug('training_traces', 'file_name')

        self.assertTrue(True)

    def test_model_remote_inference_gum_posterior_importance_sampling(self):
        samples = 2000
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_model_remote_inference_gum_posterior_inference_compilation(self):
        training_traces = 2000
        samples = 2000
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1,1], early_stop_traces=training_traces)
        posterior = self._model.posterior_distribution(samples, use_inference_network=True, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


class ModelRemoteWithReplacementTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = GaussianWithUnknownMeanMarsagliaWithReplacementCPP
        super().__init__(*args, **kwargs)

    def test_model_remote_with_replacement_prior(self):
        samples = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_remote_with_replacement_trace_length_statistics(self):
        samples = 2000
        trace_length_mean_correct = 2
        trace_length_stddev_correct = 0
        trace_length_min_correct = 2
        trace_length_max_correct = 2

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
    unittest.main(verbosity=2)
