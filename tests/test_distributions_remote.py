import unittest
import uuid
import docker
import numpy as np

import pyprob
from pyprob import util, RemoteModel
from pyprob.distributions import Empirical, Normal, Uniform, Categorical, Poisson, Bernoulli, Beta, Exponential, Gamma, LogNormal, Binomial, Weibull


# print('Pulling latest Docker image: pyprob/pyprob_cpp')
# docker_client.images.pull('pyprob/pyprob_cpp')
# print('Docker image pulled.')

class RemoteModelDistributionsTestCase(unittest.TestCase):
    def setUp(self):
        server_address = 'ipc://@RemoteModelDistributions_' + str(uuid.uuid4())
        docker_client = docker.from_env()
        self._docker_container = docker_client.containers.run('pyprob/pyprob_cpp', '/home/pyprob_cpp/build/pyprob_cpp/test_distributions {}'.format(server_address), network='host', detach=True)
        self._model = RemoteModel(server_address)

    def tearDown(self):
        self._model.close()
        self._docker_container.kill()

    def test_distributions_remote(self):
        num_samples = 4000
        prior_normal_mean_correct = Normal(1.75, 0.5).mean
        prior_uniform_mean_correct = Uniform(1.2, 2.5).mean
        prior_categorical_mean_correct = 1. # Categorical([0.1, 0.5, 0.4])
        prior_poisson_mean_correct = Poisson(4.0).mean
        prior_bernoulli_mean_correct = Bernoulli(0.2).mean
        prior_beta_mean_correct = Beta(1.2, 2.5).mean
        prior_exponential_mean_correct = Exponential(2.2).mean
        prior_gamma_mean_correct = Gamma(0.5, 1.2).mean
        prior_log_normal_mean_correct = LogNormal(0.5, 0.2).mean
        prior_binomial_mean_correct = Binomial(10, 0.72).mean
        prior_weibull_mean_correct = Weibull(1.1, 0.6).mean

        prior = self._model.prior(num_samples)
        print(prior)
        print(prior[0])
        print(prior[0].variables)
        print(prior[0].named_variables)
        prior_normal = prior.map(lambda trace: trace.named_variables['normal'].value)
        prior_uniform = prior.map(lambda trace: trace.named_variables['uniform'].value)
        prior_categorical = prior.map(lambda trace: trace.named_variables['categorical'].value)
        prior_poisson = prior.map(lambda trace: trace.named_variables['poisson'].value)
        prior_bernoulli = prior.map(lambda trace: trace.named_variables['bernoulli'].value)
        prior_beta = prior.map(lambda trace: trace.named_variables['beta'].value)
        prior_exponential = prior.map(lambda trace: trace.named_variables['exponential'].value)
        prior_gamma = prior.map(lambda trace: trace.named_variables['gamma'].value)
        prior_log_normal = prior.map(lambda trace: trace.named_variables['log_normal'].value)
        prior_binomial = prior.map(lambda trace: trace.named_variables['binomial'].value)
        prior_weibull = prior.map(lambda trace: trace.named_variables['weibull'].value)
        prior_normal_mean = util.to_numpy(prior_normal.mean)
        prior_uniform_mean = util.to_numpy(prior_uniform.mean)
        prior_categorical_mean = util.to_numpy(int(prior_categorical.mean))
        prior_poisson_mean = util.to_numpy(prior_poisson.mean)
        prior_bernoulli_mean = util.to_numpy(prior_bernoulli.mean)
        prior_beta_mean = util.to_numpy(prior_beta.mean)
        prior_exponential_mean = util.to_numpy(prior_exponential.mean)
        prior_gamma_mean = util.to_numpy(prior_gamma.mean)
        prior_log_normal_mean = util.to_numpy(prior_log_normal.mean)
        prior_binomial_mean = util.to_numpy(prior_binomial.mean)
        prior_weibull_mean = util.to_numpy(prior_weibull.mean)
        util.eval_print('num_samples', 'prior_normal_mean', 'prior_normal_mean_correct', 'prior_uniform_mean', 'prior_uniform_mean_correct', 'prior_categorical_mean', 'prior_categorical_mean_correct', 'prior_poisson_mean', 'prior_poisson_mean_correct', 'prior_bernoulli_mean', 'prior_bernoulli_mean_correct', 'prior_beta_mean', 'prior_beta_mean_correct', 'prior_exponential_mean', 'prior_exponential_mean_correct', 'prior_gamma_mean', 'prior_gamma_mean_correct', 'prior_log_normal_mean', 'prior_log_normal_mean_correct', 'prior_binomial_mean', 'prior_binomial_mean_correct', 'prior_weibull_mean', 'prior_weibull_mean_correct')

        self.assertTrue(np.allclose(prior_normal_mean, prior_normal_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_uniform_mean, prior_uniform_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_categorical_mean, prior_categorical_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_poisson_mean, prior_poisson_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_bernoulli_mean, prior_bernoulli_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_beta_mean, prior_beta_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_exponential_mean, prior_exponential_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_gamma_mean, prior_gamma_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_log_normal_mean, prior_log_normal_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_binomial_mean, prior_binomial_mean_correct, atol=0.1))
        self.assertTrue(np.allclose(prior_weibull_mean, prior_weibull_mean_correct, atol=0.1))


if __name__ == '__main__':
    # pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)

    # print('Killing Docker container {}'.format(docker_container.name))
    # docker_container.kill()
