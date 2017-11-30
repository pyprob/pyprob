#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
import numpy as np
import scipy.stats
from scipy.misc import logsumexp
import torch
import collections

class Empirical(object):
    def __init__(self, values, log_weights=None):
        self.length = len(values)
        if log_weights is None:
            log_weights = np.full(len(values), -math.log(len(values))) # assume uniform
        else:
            log_weights = np.array(log_weights)
        weights = np.exp(log_weights - logsumexp(log_weights))
        self.distribution = {}
        for i in range(self.length):
            value = values[i]
            if value in self.distribution:
                self.distribution[value] += weights[i]
            else:
                self.distribution[value] = weights[i]
        self.distribution = collections.OrderedDict(sorted(self.distribution.items()))
        self.values = np.array(list(self.distribution.keys()))
        self.weights = np.array(list(self.distribution.values()))

    def __len__(self):
        return self._length



class UniformDiscrete(object):
    def __init__(self, prior_min, prior_size):
        self.prior_min = prior_min
        self.prior_size = prior_size
        self.proposal_probabilities = None

        self.name = 'UniformDiscrete'
        self.address_suffix = '_UniformDiscrete(prior_min:{0}, prior_size:{1})'.format(self.prior_min, self.prior_size)
    def __repr__(self):
        return 'UniformDiscrete(prior_min:{0}, prior_size:{1}, proposal_probabilities:{2})'.format(self.prior_min, self.prior_size, self.proposal_probabilities)
    __str__ = __repr__
    def set_proposal_params(self, proposal_probabilities):
        self.proposal_probabilities = proposal_probabilities
    def cuda(self, device_id=None):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cuda(device_id)
    def cpu(self):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cpu()

class UniformContinuous(object):
    def __init__(self, prior_min, prior_max):
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.proposal_mode = None
        self.proposal_certainty = None

        self.name = 'UniformContinuous'
        self.address_suffix = '_UniformContinuous'
    def __repr__(self):
        return 'UniformContinuous(prior_min:{0}, prior_max:{1}, proposal_mode:{2}, proposal_certainty:{3})'.format(self.prior_min, self.prior_max, self.proposal_mode, self.proposal_certainty)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_mode_certainty):
        self.proposal_mode = tensor_of_proposal_mode_certainty[0]
        self.proposal_certainty = tensor_of_proposal_mode_certainty[1]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class UniformContinuousAlt(object):
    def __init__(self, prior_min, prior_max):
        self.prior_min = prior_min
        self.prior_max = prior_max
        self.proposal_means = None
        self.proposal_stds = None
        self.proposal_coeffs = None

        self.name = 'UniformContinuousAlt'
        self.address_suffix = '_UniformContinuousAlt'
    def __repr__(self):
        return 'UniformContinuousAlt(prior_min:{0}, prior_max:{1}, proposal_means:{2}, proposal_stds:{3}, proposal_coeffs:{4})'.format(self.prior_min, self.prior_max, self.proposal_means, self.proposal_stds, self.proposal_coeffs)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_means_stds_coeffs):
        n_components = int(tensor_of_proposal_means_stds_coeffs.size(0) / 3)
        self.proposal_means, self.proposal_stds, self.proposal_coeffs = torch.split(tensor_of_proposal_means_stds_coeffs, n_components)
    def cuda(self, device_id=None):
        if self.proposal_means is not None:
            self.proposal_means = self.proposal_means.cuda(device_id)
        if self.proposal_stds is not None:
            self.proposal_stds = self.proposal_stds.cuda(device_id)
        if self.proposal_coeffs is not None:
            self.proposal_coeffs = self.proposal_coeffs.cuda(device_id)
    def cpu(self):
        if self.proposal_means is not None:
            self.proposal_means = self.proposal_means.cpu()
        if self.proposal_stds is not None:
            self.proposal_stds = self.proposal_stds.cpu()
        if self.proposal_coeffs is not None:
            self.proposal_coeffs = self.proposal_coeffs.cpu()


class Normal(object):
    def __init__(self, prior_mean, prior_std):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.proposal_mean = None
        self.proposal_std = None

        self.name = 'Normal'
        self.address_suffix = '_Normal'
    def __repr__(self):
        return 'Normal(prior_mean:{0}, prior_std:{1}, proposal_mean:{2}, proposal_std:{3})'.format(self.prior_mean, self.prior_std, self.proposal_mean, self.proposal_std)
    __str__ = __repr__
    def sample(self):
        return np.random.normal(self.prior_mean, self.prior_std)
    def log_pdf(self, x):
        return scipy.stats.norm.logpdf(x, self.prior_mean, self.prior_std)
    def set_proposal_params(self, tensor_of_proposal_mean_std):
        self.proposal_mean = tensor_of_proposal_mean_std[0]
        self.proposal_std = tensor_of_proposal_mean_std[1]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class Flip(object):
    def __init__(self):
        self.proposal_probability = None

        self.name = 'Flip'
        self.address_suffix = '_Flip'
    def __repr__(self):
        return 'Flip(proposal_probability: {0})'.format(self.proposal_probability)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_probability):
        self.proposal_probability = tensor_of_proposal_probability[0]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class Discrete(object):
    def __init__(self, prior_size):
        self.prior_size = prior_size
        self.proposal_probabilities = None

        self.name = 'Discrete'
        self.address_suffix = '_Discrete(prior_size:{0})'.format(self.prior_size)
    def __repr__(self):
        return 'Discrete(prior_size:{0}, proposal_probabilities:{1})'.format(self.prior_size, self.proposal_probabilities)
    __str__ = __repr__
    def set_proposal_params(self, proposal_probabilities):
        self.proposal_probabilities = proposal_probabilities
    def cuda(self, device_id=None):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cuda(device_id)
    def cpu(self):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cpu()

class Categorical(object):
    def __init__(self, prior_size):
        self.prior_size = prior_size
        self.proposal_probabilities = None

        self.name = 'Categorical'
        self.address_suffix = '_Categorical(prior_size:{0})'.format(self.prior_size)
    def __repr__(self):
        return 'Categorical(prior_size:{0}, proposal_probabilities:{1})'.format(self.prior_size, self.proposal_probabilities)
    __str__ = __repr__
    def set_proposal_params(self, proposal_probabilities):
        self.proposal_probabilities = proposal_probabilities
    def cuda(self, device_id=None):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cuda(device_id)
    def cpu(self):
        if self.proposal_probabilities is not None:
            self.proposal_probabilities = self.proposal_probabilities.cpu()

class Laplace(object):
    def __init__(self, prior_location, prior_scale):
        self.prior_location = prior_location
        self.prior_scale = prior_scale
        self.proposal_location = None
        self.proposal_scale = None

        self.name = 'Laplace'
        self.address_suffix = '_Laplace'
    def __repr__(self):
        return 'Laplace(prior_location:{0}, prior_scale:{1}, proposal_location:{2}, proposal_scale:{3})'.format(self.prior_location, self.prior_scale, self.proposal_location, self.proposal_scale)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_location_scale):
        self.proposal_location = tensor_of_proposal_location_scale[0]
        self.proposal_scale = tensor_of_proposal_location_scale[1]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class Gamma(object):
    def __init__(self):
        self.proposal_location = None
        self.proposal_scale = None

        self.name = 'Gamma'
        self.address_suffix = '_Gamma'
    def __repr__(self):
        return 'Gamma(proposal_location:{0}, proposal_scale:{1})'.format(self.proposal_location, self.proposal_scale)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_location_scale):
        self.proposal_location = tensor_of_proposal_location_scale[0]
        self.proposal_scale = tensor_of_proposal_location_scale[1]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class Beta(object):
    def __init__(self):
        self.proposal_mode = None
        self.proposal_certainty = None

        self.name = 'Beta'
        self.address_suffix = '_Beta'
    def __repr__(self):
        return 'Beta(proposal_mode:{0}, proposal_certainty:{1})'.format(self.proposal_mode, self.proposal_certainty)
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_mode_certainty):
        self.proposal_mode = tensor_of_proposal_mode_certainty[0]
        self.proposal_certainty = tensor_of_proposal_mode_certainty[1]
    def cuda(self, device_id=None):
        return
    def cpu(self):
        return

class MultivariateNormal(object):
    def __init__(self, prior_mean, prior_cov):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_dim = prior_mean.size(0)
        self.proposal_mean = None
        self.proposal_vars = None

        self.name = 'MultivariateNormal'
        self.address_suffix = '_MultivariateNormal(prior_dim:{0})'.format(self.prior_dim)
    def __repr__(self):
        return 'MultivariateNormal(prior_mean:{0}, prior_cov:{1}, proposal_mean:{2}, proposal_vars:{3})'.format(self.prior_mean.numpy().tolist(), self.prior_cov.numpy().tolist(), self.proposal_mean.numpy().tolist(), self.proposal_vars.numpy().tolist())
    __str__ = __repr__
    def set_proposal_params(self, tensor_of_proposal_mean_vars):
        num_dimensions = int(tensor_of_proposal_mean_vars.size(0) / 2)
        self.proposal_mean = tensor_of_proposal_mean_vars[:num_dimensions]
        self.proposal_vars = tensor_of_proposal_mean_vars[num_dimensions:]
    def cuda(self, device_id=None):
        if self.prior_mean is not None:
            self.prior_mean = self.prior_mean.cuda(device_id)
        if self.prior_cov is not None:
            self.prior_cov = self.prior_cov.cuda(device_id)
        if self.proposal_mean is not None:
            self.proposal_mean = self.proposal_mean.cuda(device_id)
        if self.proposal_vars is not None:
            self.proposal_vars = self.proposal_vars.cuda(device_id)
    def cpu(self):
        if self.prior_mean is not None:
            self.prior_mean = self.prior_mean.cpu()
        if self.prior_cov is not None:
            self.prior_cov = self.prior_cov.cpu()
        if self.proposal_mean is not None:
            self.proposal_mean = self.proposal_mean.cpu()
        if self.proposal_vars is not None:
            self.proposal_vars = self.proposal_vars.cpu()
