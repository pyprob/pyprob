import unittest

import pyprob
from pyprob import util, state, Model
from pyprob.distributions import Categorical, Normal


class StateTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self._root_function_name = self.test_address.__code__.co_name
        super().__init__(*args, **kwargs)

    def _sample_address(self):
        address = state.extract_address(self._root_function_name)
        return address

    def test_address(self):
        address = self._sample_address()
        address_correct = '4/test_address/address'
        util.debug('address', 'address_correct')
        self.assertEqual(address, address_correct)


class PriorInflationTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class CategoricalModel(Model):
            def __init__(self):
                super().__init__('Categorical model')

            def forward(self, observation=None):
                categorical_value = pyprob.sample(Categorical([0.1, 0.1, 0.8]))
                normal_value = pyprob.sample(Normal(5., 2.))
                return float(categorical_value), normal_value

        self._model = CategoricalModel()
        super().__init__(*args, **kwargs)

    def test_prior_inflation(self):
        samples = 5000
        categorical_prior_mean_correct = 1.7
        categorical_prior_stddev_correct = 0.640312
        categorical_prior_inflated_mean_correct = 1
        categorical_prior_inflated_stddev_correct = 0.816497
        normal_prior_mean_correct = 5
        normal_prior_stddev_correct = 2
        normal_prior_inflated_mean_correct = 5
        normal_prior_inflated_stddev_correct = normal_prior_stddev_correct * 3

        prior = self._model.prior_distribution(samples, prior_inflation=pyprob.PriorInflation.DISABLED)
        categorical_prior = prior.map(lambda x: x[0])
        categorical_prior_mean = float(categorical_prior.mean)
        categorical_prior_stddev = float(categorical_prior.stddev)
        normal_prior = prior.map(lambda x: x[1])
        normal_prior_mean = float(normal_prior.mean)
        normal_prior_stddev = float(normal_prior.stddev)

        prior_inflated = self._model.prior_distribution(samples, prior_inflation=pyprob.PriorInflation.ENABLED)
        categorical_prior_inflated = prior_inflated.map(lambda x: x[0])
        categorical_prior_inflated_mean = float(categorical_prior_inflated.mean)
        categorical_prior_inflated_stddev = float(categorical_prior_inflated.stddev)
        normal_prior_inflated = prior_inflated.map(lambda x: x[1])
        normal_prior_inflated_mean = float(normal_prior_inflated.mean)
        normal_prior_inflated_stddev = float(normal_prior_inflated.stddev)

        util.debug('samples', 'categorical_prior_mean', 'categorical_prior_mean_correct', 'categorical_prior_stddev', 'categorical_prior_stddev_correct', 'categorical_prior_inflated_mean', 'categorical_prior_inflated_mean_correct', 'categorical_prior_inflated_stddev', 'categorical_prior_inflated_stddev_correct', 'normal_prior_mean', 'normal_prior_mean_correct', 'normal_prior_stddev', 'normal_prior_stddev_correct', 'normal_prior_inflated_mean', 'normal_prior_inflated_mean_correct', 'normal_prior_inflated_stddev', 'normal_prior_inflated_stddev_correct')

        self.assertAlmostEqual(categorical_prior_mean, categorical_prior_mean_correct, places=0)
        self.assertAlmostEqual(categorical_prior_stddev, categorical_prior_stddev_correct, places=0)
        self.assertAlmostEqual(categorical_prior_inflated_mean, categorical_prior_inflated_mean_correct, places=0)
        self.assertAlmostEqual(categorical_prior_inflated_stddev, categorical_prior_inflated_stddev_correct, places=0)
        self.assertAlmostEqual(normal_prior_mean, normal_prior_mean_correct, places=0)
        self.assertAlmostEqual(normal_prior_stddev, normal_prior_stddev_correct, places=0)
        self.assertAlmostEqual(normal_prior_inflated_mean, normal_prior_inflated_mean_correct, places=0)
        self.assertAlmostEqual(normal_prior_inflated_stddev, normal_prior_inflated_stddev_correct, places=0)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
