# Code from  https://github.com/dougalsutherland/opt-mmd
# Sutherland et al. "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy", ICLR 2017.
'''
Helper to perform MMD tests.

Assumes you have the feature/bigtest branch of shogun installed (including the
modular Python bindings).
'''
from __future__ import division, print_function
import os

import numpy as np
from scipy import linalg, stats
try:
    import modshogun as sg
except ImportError:  # new versions just call it shogun
    import shogun as sg

if 'OMP_NUM_THREADS' in os.environ:
    num_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    import multiprocessing as mp
    num_threads = mp.cpu_count()
sg.get_global_parallel().set_num_threads(num_threads)


def rbf_mmd_test(X, Y, bandwidth='median', null_samples=1000,
                 median_samples=1000, cache_size=32):
    '''
    Run an MMD test using a Gaussian kernel.

    Parameters
    ----------
    X : row-instance feature array

    Y : row-instance feature array

    bandwidth : float or 'median'
        The bandwidth of the RBF kernel (sigma).
        If 'median', estimates the median pairwise distance in the
        aggregate sample and uses that.

    null_samples : int
        How many times to sample from the null distribution.

    median_samples : int
        How many points to use for estimating the bandwidth.

    Returns
    -------
    p_val : float
        The obtained p value of the test.

    stat : float
        The test statistic.

    null_samples : array of length null_samples
        The samples from the null distribution.

    bandwidth : float
        The used kernel bandwidth
    '''

    if bandwidth == 'median':
        from sklearn.metrics.pairwise import euclidean_distances
        sub = lambda feats, n: feats[np.random.choice(
            feats.shape[0], min(feats.shape[0], n), replace=False)]
        Z = np.r_[sub(X, median_samples // 2), sub(Y, median_samples // 2)]
        D2 = euclidean_distances(Z, squared=True)
        upper = D2[np.triu_indices_from(D2, k=1)]
        kernel_width = np.median(upper, overwrite_input=True)
        bandwidth = np.sqrt(kernel_width / 2)
        # sigma = median / sqrt(2); works better, sometimes at least
        del Z, D2, upper
    else:
        kernel_width = 2 * bandwidth**2

    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(X.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(Y.T.astype(np.float64)))
    mmd.set_kernel(sg.GaussianKernel(cache_size, kernel_width))

    mmd.set_num_null_samples(null_samples)
    samps = mmd.sample_null()
    stat = mmd.compute_statistic()

    p_val = np.mean(stat <= samps)
    return p_val, stat, samps, bandwidth


def linear_mmd_test(X, Y, null_samples=1000):
    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(X.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(Y.T.astype(np.float64)))
    mmd.set_kernel(sg.LinearKernel())

    mmd.set_num_null_samples(null_samples)
    samps = mmd.sample_null()
    stat = mmd.compute_statistic()

    p_val = np.mean(stat <= samps)
    return p_val, stat, samps


def linear_hotelling_test(X, Y, reg=0):
    n, p = X.shape
    Z = X - Y
    Z_bar = Z.mean(axis=0)

    Z -= Z_bar
    S = Z.T.dot(Z)
    S /= (n - 1)
    if reg:
        S[::p + 1] += reg
    # z' inv(S) z = z' inv(L L') z = z' inv(L)' inv(L) z = ||inv(L) z||^2
    L = linalg.cholesky(S, lower=True, overwrite_a=True)
    Linv_Z_bar = linalg.solve_triangular(L, Z_bar, lower=True, overwrite_b=True)
    stat = n * Linv_Z_bar.dot(Linv_Z_bar)

    p_val = stats.chi2.sf(stat, p)
    return p_val, stat
