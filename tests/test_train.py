import unittest
import math
import torch
import os
import tempfile
import uuid
import shutil

import pyprob
from pyprob import util, Model, LearningRateScheduler, Optimizer
from pyprob.distributions import Normal, Uniform


class ModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMeanMarsaglia(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean (Marsaglia)')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform)
                    y = pyprob.sample(uniform)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, 0, name='obs0')
                pyprob.observe(likelihood, 0, name='obs1')
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_train_save_load_train(self):
        training_traces = 128
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=training_traces, observe_embeddings={'obs0': {'dim': 64}, 'obs1': {'dim': 64}})
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)
        self._model.learn_inference_network(num_traces=training_traces, observe_embeddings={'obs0': {'dim': 64}, 'obs1': {'dim': 64}})
        self._model.save_inference_network(file_name)
        self._model.load_inference_network(file_name)
        os.remove(file_name)
        self._model.learn_inference_network(num_traces=training_traces, observe_embeddings={'obs0': {'dim': 64}, 'obs1': {'dim': 64}})

        util.eval_print('training_traces', 'file_name')

        self.assertTrue(True)

    def test_model_save_traces_load_train(self):
        dataset_dir = tempfile.mkdtemp()
        num_traces = 512
        num_traces_per_file = 32
        training_traces = 128

        self._model.save_dataset(dataset_dir=dataset_dir, num_traces=num_traces, num_traces_per_file=num_traces_per_file)
        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=training_traces, dataset_dir=dataset_dir, batch_size=16, valid_size=16, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}})
        shutil.rmtree(dataset_dir)

        util.eval_print('dataset_dir', 'num_traces', 'num_traces_per_file', 'training_traces')

        self.assertTrue(True)

    def test_model_train(self):
        num_traces = 256

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=num_traces, batch_size=16, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}})

        util.eval_print('num_traces')

        self.assertTrue(True)

    def test_train_lr_scheduler_poly1(self):
        num_traces = 256

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=num_traces, batch_size=16, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, learning_rate_scheduler_type=LearningRateScheduler.POLY1)

        util.eval_print('num_traces')

        self.assertTrue(True)

    def test_train_lr_scheduler_poly2(self):
        num_traces = 256

        self._model.reset_inference_network()
        self._model.learn_inference_network(num_traces=num_traces, batch_size=16, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, learning_rate_scheduler_type=LearningRateScheduler.POLY2)

        util.eval_print('num_traces')

        self.assertTrue(True)

    def test_train_online_adam_larc_lr_poly2(self):
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_3 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        num_traces_end = 256
        batch_size = 16
        learning_rate_init_correct = 0.1
        learning_rate_end_correct = 0.0025

        self._model.reset_inference_network()
        print('Training\n')
        self._model.learn_inference_network(num_traces=num_traces_end/2, num_traces_end=num_traces_end, batch_size=batch_size, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, optimizer_type=Optimizer.ADAM_LARC, learning_rate_scheduler_type=LearningRateScheduler.POLY2, learning_rate_init=learning_rate_init_correct, learning_rate_end=learning_rate_end_correct, log_file_name=file_name_2)
        print('Saving\n')
        # print(self._model._inference_network._optimizer)
        optimizer_state_step_before_save = list(self._model._inference_network._optimizer.state_dict()['state'].values())[0]['step']
        self._model.save_inference_network(file_name)

        print('\nlog_file_name contents')
        with open(file_name_2) as file:
            for line in file:
                print(line, end='')

        self._model.reset_inference_network()
        print('Loading\n')
        self._model.load_inference_network(file_name)
        optimizer_state_step_after_load = list(self._model._inference_network._optimizer.state_dict()['state'].values())[0]['step']
        print('Training\n')
        self._model.learn_inference_network(num_traces=num_traces_end/2, num_traces_end=num_traces_end, batch_size=batch_size, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, optimizer_type=Optimizer.ADAM_LARC, learning_rate_scheduler_type=LearningRateScheduler.POLY2, learning_rate_init=learning_rate_init_correct, learning_rate_end=learning_rate_end_correct, log_file_name=file_name_3)
        learning_rate_end = self._model._inference_network._optimizer.param_groups[0]['lr']

        print('\nlog_file_name contents')
        with open(file_name_3) as file:
            for line in file:
                print(line, end='')

        os.remove(file_name)
        os.remove(file_name_2)
        os.remove(file_name_3)

        util.eval_print('file_name', 'file_name_2', 'file_name_3', 'num_traces_end', 'batch_size', 'learning_rate_init_correct', 'optimizer_state_step_before_save', 'optimizer_state_step_after_load', 'learning_rate_end', 'learning_rate_end_correct')
        # util.eval_print('file_name', 'file_name_2', 'file_name_3', 'num_traces_end', 'batch_size', 'learning_rate_init_correct', 'learning_rate_end', 'learning_rate_end_correct')

        self.assertEqual(optimizer_state_step_before_save, optimizer_state_step_after_load)
        self.assertAlmostEqual(learning_rate_end, learning_rate_end_correct, delta=learning_rate_end_correct)

    def test_train_offline_adam_larc_lr_poly2(self):
        dataset_dir = tempfile.mkdtemp()
        dataset_num_traces = 128
        dataset_num_traces_per_file = 32
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_2 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        file_name_3 = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        num_traces_end = 256
        batch_size = 16
        learning_rate_init_correct = 0.1
        learning_rate_end_correct = 0.0025

        print('Saving dataset\n')
        self._model.save_dataset(dataset_dir=dataset_dir, num_traces=dataset_num_traces, num_traces_per_file=dataset_num_traces_per_file)

        self._model.reset_inference_network()
        print('Training\n')
        self._model.learn_inference_network(dataset_dir=dataset_dir, num_traces=num_traces_end/2, num_traces_end=num_traces_end, batch_size=batch_size, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, optimizer_type=Optimizer.ADAM_LARC, learning_rate_scheduler_type=LearningRateScheduler.POLY2, learning_rate_init=learning_rate_init_correct, learning_rate_end=learning_rate_end_correct, log_file_name=file_name_2)
        print('Saving\n')
        # print(self._model._inference_network._optimizer)
        optimizer_state_step_before_save = list(self._model._inference_network._optimizer.state_dict()['state'].values())[0]['step']
        self._model.save_inference_network(file_name)

        print('\nlog_file_name contents')
        with open(file_name_2) as file:
            for line in file:
                print(line, end='')

        self._model.reset_inference_network()
        print('Loading\n')
        self._model.load_inference_network(file_name)
        # print(self._model._inference_network._optimizer)
        optimizer_state_step_after_load = list(self._model._inference_network._optimizer.state_dict()['state'].values())[0]['step']
        print('Training\n')
        self._model.learn_inference_network(dataset_dir=dataset_dir, num_traces=num_traces_end/2, num_traces_end=num_traces_end, batch_size=batch_size, observe_embeddings={'obs0': {'dim': 16}, 'obs1': {'dim': 16}}, optimizer_type=Optimizer.ADAM_LARC, learning_rate_scheduler_type=LearningRateScheduler.POLY2, learning_rate_init=learning_rate_init_correct, learning_rate_end=learning_rate_end_correct, log_file_name=file_name_3)
        learning_rate_end = self._model._inference_network._optimizer.param_groups[0]['lr']

        print('\nlog_file_name contents')
        with open(file_name_3) as file:
            for line in file:
                print(line, end='')

        os.remove(file_name)
        os.remove(file_name_2)
        os.remove(file_name_3)
        shutil.rmtree(dataset_dir)

        util.eval_print('dataset_dir', 'dataset_num_traces', 'dataset_num_traces_per_file', 'file_name', 'file_name_2', 'file_name_3', 'num_traces_end', 'batch_size', 'learning_rate_init_correct', 'optimizer_state_step_before_save', 'optimizer_state_step_after_load', 'learning_rate_end', 'learning_rate_end_correct')
        # util.eval_print('file_name', 'file_name_2', 'file_name_3', 'num_traces_end', 'batch_size', 'learning_rate_init_correct', 'learning_rate_end', 'learning_rate_end_correct')

        self.assertEqual(optimizer_state_step_before_save, optimizer_state_step_after_load)
        self.assertAlmostEqual(learning_rate_end, learning_rate_end_correct, delta=learning_rate_end_correct)


if __name__ == '__main__':
    pyprob.seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
