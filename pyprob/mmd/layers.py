# Code from  https://github.com/dougalsutherland/opt-mmd
# Sutherland et al. "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy", ICLR 2017.

from __future__ import division

from lasagne import init
from lasagne.layers.base import Layer
import numpy as np
import theano.tensor as T


class RBFLayer(Layer):
    '''
    An RBF network layer; output the RBF kernel value from each input to a set
    of (learned) centers.
    '''
    def __init__(self, incoming, num_centers,
                 locs=init.Normal(std=1), log_sigma=init.Constant(0.),
                 **kwargs):
        super(RBFLayer, self).__init__(incoming, **kwargs)
        self.num_centers = num_centers

        assert len(self.input_shape) == 2
        in_dim = self.input_shape[1]
        self.locs = self.add_param(locs, (num_centers, in_dim), name='locs',
                                   regularizable=False)
        self.log_sigma = self.add_param(log_sigma, (), name='log_sigma')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_centers)

    def get_output_for(self, input, **kwargs):
        gamma = 1 / (2 * T.exp(2 * self.log_sigma))

        XX = T.dot(input, input.T)
        XY = T.dot(input, self.locs.T)
        YY = T.dot(self.locs, self.locs.T)  # cache this somehow?

        X_sqnorms = T.diagonal(XX)
        Y_sqnorms = T.diagonal(YY)
        return T.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))


class SmoothedCFLayer(Layer):
    '''
    Gets the smoothed characteristic fucntion of inputs, as in eqn (14) of
    Chwialkowski et al. (NIPS 2015).

    Scales the inputs down by sigma, then tests for differences in the
    characteristic functions at locations freqs, smoothed by a Gaussian kernel
    with unit bandwidth.

    NOTE: It's *very* easy for this to get stuck with a bad log_sigma. You
    probably want to initialize it at log(median distance between inputs) or
    similar.
    '''
    def __init__(self, incoming, num_freqs,
                 freqs=init.Normal(std=1), log_sigma=init.Constant(0.),
                 **kwargs):
        super(SmoothedCFLayer, self).__init__(incoming, **kwargs)
        self.num_freqs = num_freqs

        assert len(self.input_shape) == 2
        in_dim = self.input_shape[1]
        self.freqs = self.add_param(freqs, (num_freqs, in_dim), name='freqs')
        self.log_sigma = self.add_param(log_sigma, (), name='log_sigma')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], 2 * self.num_freqs)

    def get_output_for(self, input, **kwargs):
        X = input / T.exp(self.log_sigma)
        f = T.exp(-.5 * T.sum(X ** 2, axis=1))[:, np.newaxis]
        angles = T.dot(X, self.freqs.T)
        return T.concatenate([T.sin(angles) * f, T.cos(angles) * f], axis=1)
