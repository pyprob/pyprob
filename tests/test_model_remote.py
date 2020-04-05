import torch.nn.functional as F
import unittest
import math
import time
import sys
import docker
from termcolor import colored

import pyprob
from pyprob import util, RemoteModel, InferenceEngine
from pyprob.distributions import Normal, Categorical


docker_client = docker.from_env()
print('Pulling latest Docker image: pyprob/pyprob_cpp')
docker_client.images.pull('pyprob/pyprob_cpp')
print('Docker image pulled.')

docker_container = docker_client.containers.run('pyprob/pyprob_cpp', '/home/pyprob_cpp/build/pyprob_cpp/test_set_defaults_and_addresses ipc://@RemoteModelSetDefaultsAndAddresses', network='host', detach=True)
SetDefaultsAndAddressesCPP = RemoteModel('ipc://@RemoteModelSetDefaultsAndAddresses')


class RemoteModelSetDefaultsAndAddressesTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._model = SetDefaultsAndAddressesCPP
        super().__init__(*args, **kwargs)

    def test_model_remote_set_defaults_and_addresses_prior(self):
        samples = 1000
        prior_mean_correct = 1
        prior_stddev_correct = 3.882074  # Estimate from 100k samples

        prior = self._model.prior_results(samples)
        prior_mean = float(prior.mean)
        prior_stddev = float(prior.stddev)
        util.eval_print('samples', 'prior_mean', 'prior_mean_correct', 'prior_stddev', 'prior_stddev_correct')

        self.assertAlmostEqual(prior_mean, prior_mean_correct, places=0)
        self.assertAlmostEqual(prior_stddev, prior_stddev_correct, places=0)

    def test_model_remote_set_defaults_and_addresses_addresses(self):
        addresses_controlled_correct = ['[forward()+0x252]__Normal__1', '[forward()+0x252]__Normal__2', '[forward()+0xbf1]__Normal__2']
        addresses_all_correct = ['[forward()+0x252]__Normal__1', '[forward()+0x252]__Normal__2', '[forward()+0xbf1]__Normal__1', '[forward()+0xbf1]__Normal__2', '[forward()+0x1329]__Normal__1', '[forward()+0x1329]__Normal__2', '[forward()+0x1a2e]__Normal__1']

        trace = next(self._model._trace_generator())
        addresses_controlled = [s.address for s in trace.variables_controlled]
        addresses_all = [s.address for s in trace.variables]

        util.eval_print('addresses_controlled', 'addresses_controlled_correct', 'addresses_all', 'addresses_all_correct')

        self.assertEqual(addresses_controlled, addresses_controlled_correct)
        self.assertEqual(addresses_all, addresses_all_correct)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)

    print('Killing Docker container {}'.format(docker_container.name))
    docker_container.kill()
