import pyprob
from pyprob import Model
from pyprob.distributions import Normal, Uniform

import torch
import numpy as np
import math


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

    def forward(self, observation=[]):
        mu = self.marsaglia(self.prior_mean, self.prior_stddev)
        likelihood = Normal(mu, self.likelihood_stddev)
        for o in observation:
            pyprob.observe(likelihood, o)
        return mu


def correct_posterior(x):
    p = Normal(7.25, math.sqrt(1/1.2))
    return math.exp(p.log_prob(x))

model = GaussianWithUnknownMeanMarsaglia()
model.lmh_posterior(observation=[8,9])
#  posterior_dist = model.posterior_distribution(traces=1000, observation=[8,9])
