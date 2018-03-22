import unittest
import torch
from torch.autograd import Variable

import pyprob
from pyprob import util
from pyprob.nn import ObserveEmbeddingConvNet2D5C, ObserveEmbeddingConvNet3D4C


class NNTestCase(unittest.TestCase):
    def test_ObserveEmbeddingConvNet2D5C(self):
        batch_size = 32
        channels = 3
        output_dim = 128
        input_batch_shape = [batch_size, channels, 20, 20]
        output_batch_shape_correct = [batch_size, output_dim]
        input_non_batch_shape = [channels, 20, 20]
        output_non_batch_shape_correct = [1, output_dim]

        input_batch = Variable(torch.Tensor(torch.Size(input_batch_shape)))
        input_non_batch = Variable(torch.Tensor(torch.Size(input_non_batch_shape)))
        nn = ObserveEmbeddingConvNet2D5C(input_example_non_batch=input_non_batch, output_dim=output_dim)
        nn.configure()
        output_batch_shape = list(nn.forward(input_batch).size())
        output_non_batch_shape = list(nn.forward(input_non_batch.unsqueeze(0)).size())

        util.debug('batch_size', 'channels', 'output_dim', 'input_batch_shape', 'output_batch_shape', 'output_batch_shape_correct', 'input_non_batch_shape', 'output_non_batch_shape', 'output_non_batch_shape_correct')

        self.assertEqual(output_batch_shape, output_batch_shape_correct)
        self.assertEqual(output_non_batch_shape, output_non_batch_shape_correct)

    def test_ObserveEmbeddingConvNet3D4C(self):
        batch_size = 32
        channels = 3
        output_dim = 128
        input_batch_shape = [batch_size, channels, 16, 16, 16]
        output_batch_shape_correct = [batch_size, output_dim]
        input_non_batch_shape = [channels, 16, 16, 16]
        output_non_batch_shape_correct = [1, output_dim]

        input_batch = Variable(torch.Tensor(torch.Size(input_batch_shape)))
        input_non_batch = Variable(torch.Tensor(torch.Size(input_non_batch_shape)))
        nn = ObserveEmbeddingConvNet3D4C(input_example_non_batch=input_non_batch, output_dim=output_dim)
        nn.configure()
        output_batch_shape = list(nn.forward(input_batch).size())
        output_non_batch_shape = list(nn.forward(input_non_batch.unsqueeze(0)).size())

        util.debug('batch_size', 'channels', 'output_dim', 'input_batch_shape', 'output_batch_shape', 'output_batch_shape_correct', 'input_non_batch_shape', 'output_non_batch_shape', 'output_non_batch_shape_correct')

        self.assertEqual(output_batch_shape, output_batch_shape_correct)
        self.assertEqual(output_non_batch_shape, output_non_batch_shape_correct)


if __name__ == '__main__':
    pyprob.set_verbosity(1)
    unittest.main(verbosity=2)
