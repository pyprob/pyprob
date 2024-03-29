import unittest

import pyprob
from pyprob import util, Model, InferenceEngine
from pyprob.distributions import Uniform, Normal
from pyprob.nn import OnlineDataset


class TraceTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class TestModel(Model):
            def __init__(self):
                super().__init__('Test')

            def forward(self):
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

# TODO: Commented out on 30 May 2021 after discovering that the frame.f_lasti (index of last attempted instruction in bytecode)
# in the pyprob addressing scheme changes between python versions, making these tests fail in Python 3.8, where the bytecode counters shifted by -2
# (possibly due to python compiler changes). We need to update the following tests to be more robust and not fail in these cases.
#
# class LoopTraceTestCase(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
#         class Loop(Model):
#             def __init__(self):
#                 super().__init__('Loop')
#
#             def forward(self):
#                 uniform = Uniform(-1, 1)
#                 for i in range(2):
#                     x = pyprob.sample(uniform)
#                     y = pyprob.sample(uniform)
#                     s = x*x + y*y
#
#                 likelihood = Normal(s, 0.1)
#                 pyprob.observe(likelihood, name='obs0')
#                 pyprob.observe(likelihood, name='obs1')
#                 return s
#
#         self._model = Loop()
#         super().__init__(*args, **kwargs)
#
#     def test_trace_prior(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         prior = self._model.prior(1, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
#         trace = prior[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)
#
#     def test_trace_prior_for_inference_network(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         trace = OnlineDataset(model=self._model)[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)
#
#     def test_trace_posterior_importance_sampling(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         posterior = self._model.posterior(1, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 8, 'obs1': 9})
#         trace = posterior[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)
#
#     def test_trace_posterior_importance_sampling_with_inference_network(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         self._model.learn_inference_network(num_traces=10, observe_embeddings={'obs0': {'dim': 128, 'depth': 6}, 'obs1': {'dim': 128, 'depth': 6}})
#         posterior = self._model.posterior(1, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 8, 'obs1': 9})
#         trace = posterior[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)
#
#     def test_trace_posterior_lightweight_metropolis_hastings(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         posterior = self._model.posterior(1, inference_engine=InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})
#         trace = posterior[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)
#
#     def test_trace_posterior_random_walk_metropolis_hastings(self):
#         trace_addresses_controlled_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2']
#         trace_addresses_correct = ['30__forward__x__Uniform__1', '40__forward__y__Uniform__1', '30__forward__x__Uniform__2', '40__forward__y__Uniform__2', '84__forward__?__Normal__1', '98__forward__?__Normal__1']
#
#         posterior = self._model.posterior(1, inference_engine=InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, observe={'obs0': 8, 'obs1': 9})
#         trace = posterior[0]
#         trace_addresses_controlled = [v.address for v in trace.variables_controlled]
#         trace_addresses = [v.address for v in trace.variables]
#
#         util.eval_print('trace', 'trace_addresses_controlled', 'trace_addresses_controlled_correct', 'trace_addresses', 'trace_addresses_correct')
#
#         self.assertEqual(trace_addresses_controlled_correct, trace_addresses_controlled)
#         self.assertEqual(trace_addresses_correct, trace_addresses)


if __name__ == '__main__':
    pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
