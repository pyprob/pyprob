import unittest

import pyprob
from pyprob import util


class UtilTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_util_random_seed(self):
        samples = 10

        stochastic_samples = []
        for i in range(samples):
            pyprob.set_random_seed(None)
            dist = pyprob.distributions.Normal(0, 1)
            sample = dist.sample()
            stochastic_samples.append(float(sample))

        deterministic_samples = []
        for i in range(samples):
            pyprob.set_random_seed(123)
            dist = pyprob.distributions.Normal(0, 1)
            sample = dist.sample()
            deterministic_samples.append(float(sample))

        util.eval_print('samples', 'stochastic_samples', 'deterministic_samples')
        self.assertTrue(not all(sample == stochastic_samples[0] for sample in stochastic_samples))
        self.assertTrue(all(sample == deterministic_samples[0] for sample in deterministic_samples))


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
