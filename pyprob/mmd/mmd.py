# Code from  https://github.com/dougalsutherland/opt-mmd
# Sutherland et al. "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy", ICLR 2017.
'''
MMD functions for theano variables.
For each variant, returns (mmd2, objective_to_maximize).

If you just want to call them, do e.g.:

    Xth, Yth = T.matrices('X', 'Y')
    sigmath = T.scalar('sigma')
    fn = theano.function([Xth, Yth, sigmath],
                         rbf_mmd2_and_ratio(Xth, Yth, sigma=sigmath))

    mmd2, ratio = fn(X, Y, 1)
'''
from __future__ import division
import numpy as np
import theano.tensor as T
from theano.tensor import slinalg

_eps = 1e-8

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def rbf_mmd2(X, Y, sigma=0, biased=True):
    gamma = 1 / (2 * sigma**2)

    XX = T.dot(X, X.T)
    XY = T.dot(X, Y.T)
    YY = T.dot(Y, Y.T)

    X_sqnorms = T.diagonal(XX)
    Y_sqnorms = T.diagonal(YY)

    K_XY = T.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = T.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = T.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2, mmd2


def rbf_mmd2_and_ratio(X, Y, sigma=0, biased=True):
    gamma = 1 / (2 * sigma**2)

    XX = T.dot(X, X.T)
    XY = T.dot(X, Y.T)
    YY = T.dot(Y, Y.T)

    X_sqnorms = T.diagonal(XX)
    Y_sqnorms = T.diagonal(YY)

    K_XY = T.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = T.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = T.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    return _mmd2_and_ratio(K_XX, K_XY, K_YY, unit_diagonal=True, biased=biased)


################################################################################
### Linear-time MMD with Gaussian RBF kernel

# Estimator and the idea of optimizing the ratio from:
#    Gretton, Sriperumbudur, Sejdinovic, Strathmann, and Pontil.
#    Optimal kernel choice for large-scale two-sample tests. NIPS 2012.

def rbf_mmd2_streaming(X, Y, sigma=0):
    # n = (T.smallest(X.shape[0], Y.shape[0]) // 2) * 2
    n = (X.shape[0] // 2) * 2
    gamma = 1 / (2 * sigma**2)
    rbf = lambda A, B: T.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()
    return mmd2, mmd2


def rbf_mmd2_streaming_and_ratio(X, Y, sigma=0):
    # n = (T.smallest(X.shape[0], Y.shape[0]) // 2) * 2
    n = (X.shape[0] // 2) * 2
    gamma = 1 / (2 * sigma**2)
    rbf = lambda A, B: T.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    h_bits = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
            - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2]))

    mmd2 = h_bits.mean()

    # variance is 1/2 E_{v, v'} (h(v) - h(v'))^2
    # estimate with even, odd diffs
    m = (n // 2) * 2
    approx_var = 1/2 * ((h_bits[:m:2] - h_bits[1:m:2]) ** 2).mean()
    ratio = mmd2 / T.sqrt(T.largest(approx_var, _eps))
    return mmd2, ratio


################################################################################
### MMD with linear kernel

# Hotelling test statistic is from:
#    Jitkrittum, Szabo, Chwialkowski, and Gretton.
#    Interpretable Distribution Features with Maximum Testing Power.
#    NIPS 2015.

def linear_mmd2(X, Y, biased=True):
    if not biased:
        raise ValueError("Haven't implemented unbiased linear_mmd2 yet")
    X_bar = X.mean(axis=0)
    Y_bar = Y.mean(axis=0)
    Z_bar = X_bar - Y_bar
    mmd2 = Z_bar.dot(Z_bar)
    return mmd2, mmd2


def linear_mmd2_and_hotelling(X, Y, biased=True, reg=0):
    if not biased:
        raise ValueError("linear_mmd2_and_hotelling only works for biased est")

    n = X.shape[0]
    p = X.shape[1]
    Z = X - Y
    Z_bar = Z.mean(axis=0)
    mmd2 = Z_bar.dot(Z_bar)

    Z_cent = Z - Z_bar
    S = Z_cent.T.dot(Z_cent) / (n - 1)
    # z' inv(S) z = z' inv(L L') z = z' inv(L)' inv(L) z = ||inv(L) z||^2
    L = slinalg.cholesky(S + reg * T.eye(p))
    Linv_Z_bar = slinalg.solve_lower_triangular(L, Z_bar)
    lambda_ = n * Linv_Z_bar.dot(Linv_Z_bar)
    # happens on the CPU!
    return mmd2, lambda_


def linear_mmd2_and_ratio(X, Y, biased=True):
    # TODO: can definitely do this faster for a linear kernel...
    K_XX = T.dot(X, X.T)
    K_XY = T.dot(X, Y.T)
    K_YY = T.dot(Y, Y.T)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, unit_diagonal=False, biased=biased)


################################################################################
### Helper functions to compute variances based on kernel matrices

def _mmd2_and_ratio(K_XX, K_XY, K_YY, unit_diagonal=False, biased=False,
                    min_var_est=_eps):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, unit_diagonal=unit_diagonal, biased=biased)
    ratio = mmd2 / T.sqrt(T.largest(var_est, min_var_est))
    return mmd2, ratio


def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False, biased=False):
    m = K_XX.shape[0]  # Assumes X, Y are same shape

    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = T.diagonal(K_XX)
        diag_Y = T.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum  = (K_XY ** 2).sum()

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m-1))
              + Kt_YY_sum / (m * (m-1))
              - 2 * K_XY_sum / (m * m))

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              K_XY_sums_1.dot(K_XY_sums_1)
            + K_XY_sums_0.dot(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
    )

    return mmd2, var_est
