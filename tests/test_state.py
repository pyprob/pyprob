import unittest
import math

import pyprob
from pyprob import util, state, Model, InferenceEngine
from pyprob.distributions import Categorical, Normal

importance_sampling_samples = 4000
lightweight_metropolis_hastings_samples = 7500
lightweight_metropolis_hastings_burn_in = 500


class StateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._root_function_name = self.test_state_address.__code__.co_name
        super().__init__(*args, **kwargs)

    def _sample_address(self):
        address = state._extract_address(self._root_function_name)
        return address

    def test_state_address(self):
        address = self._sample_address()
        address_correct = '4__test_state_address__address'
        util.eval_print('address', 'address_correct')
        self.assertEqual(address, address_correct)


class FactorTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # See "3.2.1 Conditioning with Factors" in van de Meent, J.W., Paige, B., Yang, H. and Wood, F., 2018. An introduction to probabilistic programming. arXiv preprint arXiv:1809.10756.
        class FactorModel(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean using factor')

            def forward(self):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                likelihood_func = lambda x: likelihood.log_prob(x)
                pyprob.factor(log_prob_func=likelihood_func, name='obs0')
                pyprob.factor(log_prob_func=likelihood_func, name='obs1')
                return mu

        class FactorModel2(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean using factor')

            def forward(self):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                likelihood_func = lambda x: likelihood.log_prob(x)
                pyprob.factor(log_prob=likelihood_func(8))
                pyprob.factor(log_prob=likelihood_func(9))
                return mu

        self._model = FactorModel()
        self._model2 = FactorModel2()
        super().__init__(*args, **kwargs)

    def test_factor_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.005

        posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, delta=0.75)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.33)

    def test_factor_gum_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        posterior = self._model.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})[burn_in:]

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertLess(kl_divergence, 0.33)

    def test_factor2_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)
        posterior_effective_sample_size_min = samples * 0.005

        posterior = self._model2.posterior_results(samples, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        posterior_effective_sample_size = float(posterior.effective_sample_size)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'posterior_effective_sample_size', 'posterior_effective_sample_size_min', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, delta=0.75)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertGreater(posterior_effective_sample_size, posterior_effective_sample_size_min)
        self.assertLess(kl_divergence, 0.33)

    def test_factor2_gum_posterior_lightweight_metropolis_hastings(self):
        samples = lightweight_metropolis_hastings_samples
        burn_in = lightweight_metropolis_hastings_burn_in
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)

        posterior = self._model2.posterior_results(samples, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS)[burn_in:]

        posterior_mean = float(posterior.mean)
        posterior_stddev = float(posterior.stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.eval_print('samples', 'burn_in', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, delta=0.75)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, delta=0.75)
        self.assertLess(kl_divergence, 0.33)


class PriorInflationTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class CategoricalModel(Model):
            def __init__(self):
                super().__init__('Categorical model')

            def forward(self):
                categorical_value = pyprob.sample(Categorical([0.1, 0.1, 0.8]))
                normal_value = pyprob.sample(Normal(5., 2.))
                return float(categorical_value), normal_value

        self._model = CategoricalModel()
        super().__init__(*args, **kwargs)

    def test_state_prior_inflation(self):
        samples = 5000
        categorical_prior_mean_correct = 1.7
        categorical_prior_stddev_correct = 0.640312
        categorical_prior_inflated_mean_correct = 1
        categorical_prior_inflated_stddev_correct = 0.816497
        normal_prior_mean_correct = 5
        normal_prior_stddev_correct = 2
        normal_prior_inflated_mean_correct = 5
        normal_prior_inflated_stddev_correct = normal_prior_stddev_correct * 3

        prior = self._model.prior_results(samples, prior_inflation=pyprob.PriorInflation.DISABLED)
        categorical_prior = prior.map(lambda x: x[0])
        categorical_prior_mean = float(categorical_prior.mean)
        categorical_prior_stddev = float(categorical_prior.stddev)
        normal_prior = prior.map(lambda x: x[1])
        normal_prior_mean = float(normal_prior.mean)
        normal_prior_stddev = float(normal_prior.stddev)

        prior_inflated = self._model.prior_results(samples, prior_inflation=pyprob.PriorInflation.ENABLED)
        categorical_prior_inflated = prior_inflated.map(lambda x: x[0])
        categorical_prior_inflated_mean = float(categorical_prior_inflated.mean)
        categorical_prior_inflated_stddev = float(categorical_prior_inflated.stddev)
        normal_prior_inflated = prior_inflated.map(lambda x: x[1])
        normal_prior_inflated_mean = float(normal_prior_inflated.mean)
        normal_prior_inflated_stddev = float(normal_prior_inflated.stddev)

        util.eval_print('samples', 'categorical_prior_mean', 'categorical_prior_mean_correct', 'categorical_prior_stddev', 'categorical_prior_stddev_correct', 'categorical_prior_inflated_mean', 'categorical_prior_inflated_mean_correct', 'categorical_prior_inflated_stddev', 'categorical_prior_inflated_stddev_correct', 'normal_prior_mean', 'normal_prior_mean_correct', 'normal_prior_stddev', 'normal_prior_stddev_correct', 'normal_prior_inflated_mean', 'normal_prior_inflated_mean_correct', 'normal_prior_inflated_stddev', 'normal_prior_inflated_stddev_correct')

        self.assertAlmostEqual(categorical_prior_mean, categorical_prior_mean_correct, places=0)
        self.assertAlmostEqual(categorical_prior_stddev, categorical_prior_stddev_correct, places=0)
        self.assertAlmostEqual(categorical_prior_inflated_mean, categorical_prior_inflated_mean_correct, places=0)
        self.assertAlmostEqual(categorical_prior_inflated_stddev, categorical_prior_inflated_stddev_correct, places=0)
        self.assertAlmostEqual(normal_prior_mean, normal_prior_mean_correct, places=0)
        self.assertAlmostEqual(normal_prior_stddev, normal_prior_stddev_correct, places=0)
        self.assertAlmostEqual(normal_prior_inflated_mean, normal_prior_inflated_mean_correct, places=0)
        self.assertAlmostEqual(normal_prior_inflated_stddev, normal_prior_inflated_stddev_correct, places=0)


if __name__ == '__main__':
    pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
