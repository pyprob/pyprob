import pyprob
from pyprob.nn import ProposalUniformTruncatedNormalMixture
from pyprob.trace import Variable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from termcolor import colored


# Inspired by Thomas Viehmann
# https://github.com/t-vi/pytorch-tvmisc/blob/master/misc/Mixture_Density_Network_Gaussian_1d.ipynb
class TestDataset(Dataset):
    def __init__(self, n):
        self._n = n
        self._y = torch.rand(n) * 21 - 10.5
        self._x = torch.sin(0.75 * self._y) * 7 + self._y * 0.5 + torch.randn(n)
        self._y = torch.clamp(self._y, min=-10, max=10)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return (self._x[i], self._y[i])


def produce_results(results_dir):
    training_data = 1024
    training_epochs = 2000
    batch_size = 32
    test_iters = 512

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    pyprob.util.create_path(results_dir, directory=True)

    dataset = TestDataset(training_data)
    net = ProposalUniformTruncatedNormalMixture(input_shape=[1], output_shape=[1], num_layers=2, mixture_components=10)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    print('Module: {}'.format(net))

    print('Training')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    losses = []
    for epoch in range(training_epochs):
        loss_epoch = 0
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            dist = net.forward(x, [Variable(pyprob.distributions.Uniform(low=-10, high=10))]*batch_size)
            loss = -dist.log_prob(y, sum=True)
            loss_epoch += float(loss)
            loss.backward()
            optimizer.step()
        loss_epoch /= len(dataset)
        losses.append(loss_epoch)
        print('Epoch: {}, loss: {:+.4e}'.format(epoch, loss_epoch))

    print('Testing')
    test_dataset = TestDataset(test_iters)
    test_x = []
    test_y = []
    for i in range(test_iters):
        x = test_dataset._x[i]
        dist = net.forward(x.unsqueeze(0), [Variable(pyprob.distributions.Uniform(low=-10, high=10))])
        for i in range(10):
            y = dist.sample()
            test_x.append(float(x))
            test_y.append(float(y))
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    plot_file_name = os.path.join(results_dir, 'result.pdf')
    print('Saving result plot to: {}'.format(plot_file_name))
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(dataset._x.numpy(), dataset._y.numpy(), s=2, label='train')
    plt.scatter(test_x, test_y, s=2, label='test')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig(plot_file_name)

    loss_plot_file_name = os.path.join(results_dir, 'loss.pdf')
    print('Saving loss plot to: {}'.format(loss_plot_file_name))
    fig = plt.figure(figsize=(8, 8))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(loss_plot_file_name)


if __name__ == '__main__':
    pyprob.set_random_seed(1)
    pyprob.set_cuda(False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('Current dir: {}'.format(current_dir))

    results_dir = os.path.join(current_dir, 'proposal_uniform_truncated_normal_mixture')
    produce_results(results_dir=results_dir)

    print('Done')
