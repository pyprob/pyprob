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
                ret = pyprob.sample(uniform)
                ret = pyprob.sample(uniform)
                ret = pyprob.sample(uniform, control=False)
                ret = pyprob.sample(uniform, control=False)
                ret = pyprob.sample(uniform, control=False)
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                pyprob.observe(uniform, 0.5)
                return ret

        self._model = TestModel()
        super().__init__(*args, **kwargs)

    def test_trace_controlled_uncontrolled_observed(self):
        controlled_correct = 2
        uncontrolled_correct = 3
        observed_correct = 4

        trace = self._model._prior_traces(1)[0]
        controlled = len(trace.samples)
        uncontrolled = len(trace.samples_uncontrolled)
        observed = len(trace.samples_observed)

        util.debug('controlled', 'controlled_correct', 'uncontrolled', 'uncontrolled_correct', 'observed', 'observed_correct')

        self.assertEqual(controlled, controlled_correct)
        self.assertEqual(uncontrolled, uncontrolled_correct)
        self.assertEqual(observed, observed_correct)

    def test_trace_save_trace_cache_train(self):
        cache_files = 1
        cache_traces_per_file = 1024
        training_traces = 128
        path_name = tempfile.mkdtemp()

        self._model.use_trace_cache(path_name)
        self._model.save_trace_cache(path_name, files=cache_files, traces_per_file=cache_traces_per_file, observation=[0, 0])
        self._model.learn_inference_network(observation=[0, 0], num_traces=training_traces, use_trace_cache=True)
        shutil.rmtree(path_name)

        util.debug('path_name', 'cache_files', 'cache_traces_per_file', 'training_traces')

        self.assertTrue(True)


if __name__ == '__main__':
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
