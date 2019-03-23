import unittest
import shutil
import tempfile

import pyprob
from pyprob import util
from pyprob import Model
from pyprob.distributions import Uniform


class TraceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class TestModel(Model):
            def __init__(self):
                super().__init__('Test')

            def forward(self, observation=[]):
                uniform = Uniform(0, 1)
                val = pyprob.sample(uniform)
                val = pyprob.sample(uniform)
                val = pyprob.sample(uniform, control=False)
                val = pyprob.sample(uniform, control=False)
                val = pyprob.sample(uniform, control=False)
                pyprob.tag(value=val, name='val')
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                return val

        self._model = TestModel()
        super().__init__(*args, **kwargs)

    def test_trace_controlled_uncontrolled_observed(self):
        controlled_correct = 2
        uncontrolled_correct = 3
        observed_correct = 4

        trace = self._model._traces(1)[0]
        controlled = len(trace.variables_controlled)
        uncontrolled = len(trace.variables_uncontrolled)
        observed = len(trace.variables_observed)
        tagged_val = 'val' in trace.named_variables

        util.eval_print('controlled', 'controlled_correct', 'uncontrolled', 'uncontrolled_correct', 'observed', 'observed_correct', 'tagged_val')

        self.assertEqual(controlled, controlled_correct)
        self.assertEqual(uncontrolled, uncontrolled_correct)
        self.assertEqual(observed, observed_correct)
        self.assertTrue(tagged_val)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
