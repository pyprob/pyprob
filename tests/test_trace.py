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
                pyprob.observe(value=val, name='val')
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
        observed_correct = 5

        trace = self._model._traces(1)[0][0]
        controlled = len(trace.variables_controlled)
        uncontrolled = len(trace.variables_uncontrolled)
        observed = len(trace.variables_observed)
        observed_val = 'val' in trace.named_variables

        util.debug('controlled', 'controlled_correct', 'uncontrolled', 'uncontrolled_correct', 'observed', 'observed_correct', 'observed_val')

        self.assertEqual(controlled, controlled_correct)
        self.assertEqual(uncontrolled, uncontrolled_correct)
        self.assertEqual(observed, observed_correct)
        self.assertTrue(observed_val)

    # def test_trace_save_trace_cache_train(self):
    #     cache_files = 4
    #     cache_traces_per_file = 128
    #     training_traces = 128
    #     path_name = tempfile.mkdtemp()
    #
    #     self._model.use_trace_cache(path_name)
    #     self._model.save_trace_cache(path_name, files=cache_files, traces_per_file=cache_traces_per_file, observation=[0, 0])
    #     self._model.learn_inference_network(observation=[0, 0], num_traces=training_traces, use_trace_cache=True, batch_size=64, valid_size=256)
    #     shutil.rmtree(path_name)
    #
    #     util.debug('path_name', 'cache_files', 'cache_traces_per_file', 'training_traces')
    #
    #     self.assertTrue(True)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
