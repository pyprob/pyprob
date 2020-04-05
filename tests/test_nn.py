import unittest
import torch

import pyprob
from pyprob import util
from pyprob.nn import EmbeddingFeedForward, EmbeddingCNN2D5C, EmbeddingCNN3D5C


class NNTestCase(unittest.TestCase):
    def test_nn_EmbeddingFeedForward(self):
        batch_size = 32
        input_shape = [100, 100]
        output_shape = [128]
        input_batch_shape = [batch_size] + input_shape
        output_batch_shape_correct = [batch_size] + output_shape

        input_batch = torch.zeros(input_batch_shape)
        nn = EmbeddingFeedForward(input_shape=torch.Size(input_shape), output_shape=torch.Size(output_shape))
        output_batch = nn(input_batch)
        output_batch_shape = list(output_batch.size())

        util.eval_print('input_shape', 'output_shape', 'batch_size', 'input_batch_shape', 'output_batch_shape', 'output_batch_shape_correct')

        self.assertEqual(output_batch_shape, output_batch_shape_correct)

    def test_nn_EmbeddingCNN2D5C(self):
        batch_size = 32
        input_shape = [3, 100, 100]
        output_shape = [128]
        input_batch_shape = [batch_size] + input_shape
        output_batch_shape_correct = [batch_size] + output_shape

        input_batch = torch.zeros(input_batch_shape)
        nn = EmbeddingCNN2D5C(input_shape=torch.Size(input_shape), output_shape=torch.Size(output_shape))
        output_batch = nn(input_batch)
        output_batch_shape = list(output_batch.size())

        util.eval_print('input_shape', 'output_shape', 'batch_size', 'input_batch_shape', 'output_batch_shape', 'output_batch_shape_correct')

        self.assertEqual(output_batch_shape, output_batch_shape_correct)

    def test_nn_EmbeddingCNN3D5C(self):
        batch_size = 32
        input_shape = [2, 25, 25, 25]
        output_shape = [128]
        input_batch_shape = [batch_size] + input_shape
        output_batch_shape_correct = [batch_size] + output_shape

        input_batch = torch.zeros(input_batch_shape)
        nn = EmbeddingCNN3D5C(input_shape=torch.Size(input_shape), output_shape=torch.Size(output_shape))
        output_batch = nn(input_batch)
        output_batch_shape = list(output_batch.size())

        util.eval_print('input_shape', 'output_shape', 'batch_size', 'input_batch_shape', 'output_batch_shape', 'output_batch_shape_correct')

        self.assertEqual(output_batch_shape, output_batch_shape_correct)


if __name__ == '__main__':
    pyprob.set_random_seed(123)
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
