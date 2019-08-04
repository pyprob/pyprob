import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import util


class Distribution():
    def __init__(self, name, address_suffix='', batch_shape=torch.Size(), event_shape=torch.Size(), torch_dist=None):
        self.name = name
        self._address_suffix = address_suffix
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._torch_dist = torch_dist

    @property
    def batch_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.batch_shape
        else:
            return self._batch_shape

    @property
    def event_shape(self):
        if self._torch_dist is not None:
            return self._torch_dist.event_shape
        else:
            return self._event_shape

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()

    def log_prob(self, value, sum=False):
        if self._torch_dist is not None:
            lp = self._torch_dist.log_prob(util.to_tensor(value))
            return torch.sum(lp) if sum else lp
        else:
            raise NotImplementedError()

    def prob(self, value):
        return torch.exp(self.log_prob(util.to_tensor(value)))

    def plot(self, min_val=-10, max_val=10, step_size=0.1, figsize=(10, 5), xlabel=None, ylabel='Probability', xticks=None, yticks=None, log_xscale=False, log_yscale=False, file_name=None, show=True, fig=None, *args, **kwargs):
        if fig is None:
            if not show:
                mpl.rcParams['axes.unicode_minus'] = False
                plt.switch_backend('agg')
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
        xvals = np.arange(min_val, max_val, step_size)
        plt.plot(xvals, [torch.exp(self.log_prob(x)) for x in xvals], *args, **kwargs)
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        # if xlabel is None:
        #     xlabel = self.name
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if file_name is not None:
            plt.savefig(file_name)
        if show:
            plt.show()

    @property
    def mean(self):
        if self._torch_dist is not None:
            return self._torch_dist.mean
        else:
            raise NotImplementedError()

    @property
    def variance(self):
        if self._torch_dist is not None:
            return self._torch_dist.variance
        else:
            raise NotImplementedError()

    @property
    def stddev(self):
        return self.variance.sqrt()

    def expectation(self, func):
        raise NotImplementedError()

    @staticmethod
    def kl_divergence(distribution_1, distribution_2):
        if distribution_1._torch_dist is None or distribution_2._torch_dist is None:
            raise ValueError('KL divergence is not currently supported for this pair of distributions.')
        return torch.distributions.kl.kl_divergence(distribution_1._torch_dist, distribution_2._torch_dist)
