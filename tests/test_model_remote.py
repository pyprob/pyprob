import unittest
import math
import uuid
import tempfile
import os
import docker
import torch
import torch.nn.functional as F

import pyprob
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

docker_client.containers.run('probprog/cpproblight', '/code/cpproblight/build/cpproblight/test_hmm tcp://*:5557', network='host', detach=True)
HiddenMarkovModelCPP = ModelRemote('tcp://127.0.0.1:5557')

docker_client.containers.run('probprog/cpproblight', '/code/cpproblight/build/cpproblight/test_set_defaults_and_addresses tcp://*:5558', network='host', detach=True)
SetDefaultsAndAddressesCPP = ModelRemote('tcp://127.0.0.1:5558')


class ModelRemoteGaussianWithUnknownMeanMarsagliaTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = GaussianWithUnknownMeanMarsagliaCPP
        super().__init__(*args, **kwargs)

    def test_model_remote_gum_marsaglia_prior(self):
        samples = 5000
        prior_mean_correct = 1
        prior_stddev_correct = math.sqrt(5)

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')
        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_remote_gum_marsaglia_trace_length_statistics(self):
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

    def test_model_remote_gum_marsaglia_train_save_load(self):
        training_traces = 128
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        self._model.learn_inference_network(observation=[1, 1], num_traces=training_traces)
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)

        util.debug('training_traces', 'file_name')

        self.assertTrue(True)

    def test_model_remote_gum_marsaglia_inference_gum_posterior_importance_sampling(self):
        samples = 2000
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)

    def test_model_remote_gum_marsaglia_inference_gum_posterior_inference_compilation(self):
        training_traces = 2000
        samples = 2000
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1,1], num_traces=training_traces)
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
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


class ModelRemoteHiddenMarkovModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = HiddenMarkovModelCPP

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

    def test_model_remote_hmm_posterior_importance_sampling(self):
        samples = 500

        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance')

        self.assertLess(l2_distance, 10)

    def test_model_remote_hmm_posterior_inference_compilation(self):
        training_traces = 500
        samples = 500

        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        self._model.learn_inference_network(observation=torch.zeros(16), num_traces=training_traces)
        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observation=observation)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance')

        self.assertLess(l2_distance, 10)

    def test_model_remote_hmm_posterior_metropolis_hastings(self):
        samples = 1000

        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        posterior = self._model.posterior_distribution(samples, inference_engine=pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observation=observation)
        posterior_mean_unweighted = posterior.unweighted().mean
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance')

        self.assertLess(l2_distance, 10)


class ModelRemoteSetDefaultsAndAddressesTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = SetDefaultsAndAddressesCPP
        super().__init__(*args, **kwargs)

    def test_model_remote_set_defaults_and_addresses_prior(self):
        samples = 2000
        prior_mean_correct = 1
        prior_stddev_correct = 3.882074  # Estimate from 100k samples

        prior = self._model.prior_distribution(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.debug('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_remote_set_defaults_and_addresses_addresses(self):
        addresses_correct = ['normal1_Normal_1', 'normal1_Normal_2', 'normal2_Normal_replaced']
        addresses_all_correct = ['normal1_Normal_1', 'normal1_Normal_2', 'normal2_Normal_replaced', 'normal2_Normal_replaced', 'normal3_Normal_1', 'normal3_Normal_2', 'likelihood_Normal_1']

        trace = next(self._model._prior_trace_generator(observation=[0]))
        addresses = [s.address for s in trace.samples]
        addresses_all = [s.address for s in trace._samples_all]

        util.debug('addresses', 'addresses_correct', 'addresses_all', 'addresses_all_correct')

        self.assertEqual(addresses, addresses_correct)
        self.assertEqual(addresses_all, addresses_all_correct)


if __name__ == '__main__':
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
