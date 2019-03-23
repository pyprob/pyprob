import torch

from . import Distribution, Categorical
from .. import util


class Mixture(Distribution):
    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length = len(distributions)
        if probs is None:
            self._probs = util.to_tensor(torch.zeros(self.length)).fill_(1./self.length)
        else:
            self._probs = util.to_tensor(probs)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        self._log_probs = torch.log(util.clamp_probs(self._probs))

        event_shape = torch.Size()
        if self._probs.dim() == 1:
            batch_shape = torch.Size()
            self._batch_length = 0
        elif self._probs.dim() == 2:
            batch_shape = torch.Size([self._probs.size(0)])
            self._batch_length = self._probs.size(0)
        else:
            raise ValueError('Expecting a 1d or 2d (batched) mixture probabilities.')
        self._mixing_dist = Categorical(self._probs)
        self._mean = None
        self._variance = None
        super().__init__(name='Mixture', address_suffix='Mixture({})'.format(', '.join([d._address_suffix for d in self._distributions])), batch_shape=batch_shape, event_shape=event_shape)

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value, sum=False):
        if self._batch_length == 0:
            value = util.to_tensor(value).squeeze()
            lp = torch.logsumexp(self._log_probs + util.to_tensor([d.log_prob(value) for d in self._distributions]), dim=0)
        else:
            value = util.to_tensor(value).view(self._batch_length)
            lp = torch.logsumexp(self._log_probs + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).view(-1, self._batch_length).t(), dim=1)
        return torch.sum(lp) if sum else lp

    def sample(self):
        if self._batch_length == 0:
            i = int(self._mixing_dist.sample())
            return self._distributions[i].sample()
        else:
            indices = self._mixing_dist.sample()
            dist_samples = []
            for d in self._distributions:
                sample = d.sample()
                if sample.dim() == 0:
                    sample = sample.unsqueeze(-1)
                dist_samples.append(sample)
            ret = []
            for b in range(self._batch_length):
                i = int(indices[b])
                ret.append(dist_samples[i][b])
            return util.to_tensor(ret)

    @property
    def mean(self):
        if self._mean is None:
            means = torch.stack([d.mean for d in self._distributions])
            if self._batch_length == 0:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = torch.stack([(d.mean - self.mean).pow(2) + d.variance for d in self._distributions])
            if self._batch_length == 0:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
        return self._variance
