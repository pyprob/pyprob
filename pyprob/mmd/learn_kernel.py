# Code from  https://github.com/dougalsutherland/opt-mmd
# Sutherland et al. "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy", ICLR 2017.

from __future__ import division, print_function

import ast
import os
import sys
import tempfile
import time
import types

import numpy as np
import lasagne
from six import exec_
from sklearn.metrics.pairwise import euclidean_distances
import theano
import theano.tensor as T

from . import mmd
from . import mmd_test


floatX = np.dtype(theano.config.floatX)
def make_floatX(x):
    return np.array(x, dtype=floatX)[()]


################################################################################
################################################################################
### Making the representation network


def test_func(get_rep, X_test, Y_test, sigma, args):
    print("Testing...", end='')
    sys.stdout.flush()
    try:
        p_val, stat, null_samps = eval_rep(
            get_rep, X_test, Y_test,
            linear_kernel=args.linear_kernel, sigma=sigma,
            hotelling=args.criterion == 'hotelling',
            null_samples=args.null_samples)
        print("p-value: {}".format(p_val))
    except ImportError as e:
        print()
        print("Couldn't import shogun:\n{}".format(e), file=sys.stderr)
        p_val, stat, null_samps = None, None, None

    return p_val, stat, null_samps


def net_nothing(net_p, net_q):
    return net_p, net_q, 0


def net_scaling(net_p, net_q):
    net_p = lasagne.layers.ScaleLayer(net_p)
    net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
    return net_p, net_q, 0


def net_scaling_exp(net_p, net_q):
    log_scales = theano.shared(np.zeros(net_p.output_shape[1], floatX),
                               name='log_scales')
    net_p = lasagne.layers.ScaleLayer(net_p, scales=T.exp(log_scales))
    net_q = lasagne.layers.ScaleLayer(net_q, scales=net_p.scales)
    return net_p, net_q, 0


def net_rbf(net_p, net_q, J=5):
    '''
    Network equivalent to Wittawat's mean embedding test:
    compute RBF kernel values to each of J test points.
    '''
    from layers import RBFLayer
    net_p = RBFLayer(net_p, J)
    net_q = RBFLayer(net_q, J, locs=net_p.locs, log_sigma=net_p.log_sigma)
    return net_p, net_q, 0


def net_scf(net_p, net_q, n_freqs=5):
    '''
    Network equivalent to Wittawat's smoothed characteristic function test.
    '''
    from layers import SmoothedCFLayer
    net_p = SmoothedCFLayer(net_p, n_freqs)
    net_q = SmoothedCFLayer(net_q, n_freqs,
                            freqs=net_p.freqs, log_sigma=net_p.log_sigma)
    return net_p, net_q, 0


def _paired_dense(in_1, in_2, **kwargs):
    d_1 = lasagne.layers.DenseLayer(in_1, **kwargs)
    d_2 = lasagne.layers.DenseLayer(in_2, W=d_1.W, b=d_1.b, **kwargs)
    return d_1, d_2


def net_basic(net_p, net_q):
    net_p, net_q = _paired_dense(
        net_p, net_q, num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify)
    net_p, net_q = _paired_dense(
        net_p, net_q, num_units=64,
        nonlinearity=lasagne.nonlinearities.rectify)
    return net_p, net_q, 0


net_versions = {
    'nothing': net_nothing,
    'scaling': net_scaling,
    'scaling-exp': net_scaling_exp,
    'rbf': net_rbf,
    'scf': net_scf,
    'basic': net_basic,
}


def register_custom_net(code):
    module = types.ModuleType('net_custom', 'Custom network function')
    exec_(code, module.__dict__)
    sys.modules['net_custom']= module
    net_versions['custom'] = module.net_custom


################################################################################
### Adding loss and so on to the network

def make_network(input_p, input_q, dim,
                 criterion='mmd', biased=True, streaming_est=False,
                 linear_kernel=False, log_sigma=0, hotelling_reg=0,
                 opt_log=True, batchsize=None,
                 net_version='nothing'):

    in_p = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_p)
    in_q = lasagne.layers.InputLayer(shape=(batchsize, dim), input_var=input_q)
    net_p, net_q, reg = net_versions[net_version](in_p, in_q)
    rep_p, rep_q = lasagne.layers.get_output([net_p, net_q])

    choices = {  # criterion, linear kernel, streaming
        ('mmd', False, False): mmd.rbf_mmd2,
        ('mmd', False, True): mmd.rbf_mmd2_streaming,
        ('mmd', True, False): mmd.linear_mmd2,
        ('ratio', False, False): mmd.rbf_mmd2_and_ratio,
        ('ratio', False, True): mmd.rbf_mmd2_streaming_and_ratio,
        ('ratio', True, False): mmd.linear_mmd2_and_ratio,
        ('hotelling', True, False): mmd.linear_mmd2_and_hotelling,
    }
    try:
        fn = choices[criterion, linear_kernel, streaming_est]
    except KeyError:
        raise ValueError("Bad parameter combo: criterion = {}, {}, {}".format(
            criterion,
            "linear kernel" if linear_kernel else "rbf kernel",
            "streaming" if streaming_est else "not streaming"))

    kwargs = {}
    if linear_kernel:
        log_sigma = None
    else:
        log_sigma = theano.shared(make_floatX(log_sigma), name='log_sigma')
        kwargs['sigma'] = T.exp(log_sigma)
    if not streaming_est:
        kwargs['biased'] = biased
    if criterion == 'hotelling':
        kwargs['reg'] = hotelling_reg

    mmd2_pq, stat = fn(rep_p, rep_q, **kwargs)
    obj = -(T.log(T.largest(stat, 1e-6)) if opt_log else stat) + reg
    return mmd2_pq, obj, rep_p, net_p, net_q, log_sigma


################################################################################
### Training helpers

def iterate_minibatches(*arrays, **kwds):
    batchsize = kwds['batchsize']
    shuffle = kwds.get('shuffle', False)

    assert len(arrays) > 0
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays[1:])

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)

    for start_idx in range(0, max(0, n - batchsize) + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(a[excerpt] for a in arrays)


def run_train_epoch(X_train, Y_train, batchsize, train_fn):
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    batches = zip( # shuffle the two independently
        iterate_minibatches(X_train, batchsize=batchsize, shuffle=True),
        iterate_minibatches(Y_train, batchsize=batchsize, shuffle=True),
    )
    for ((Xbatch,), (Ybatch,)) in batches:
        mmd2, obj = train_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    return total_mmd2 / n_batches, total_obj / n_batches


def run_val(X_val, Y_val, batchsize, val_fn):
    total_mmd2 = 0
    total_obj = 0
    n_batches = 0
    for (Xbatch, Ybatch) in iterate_minibatches(
                X_val, Y_val, batchsize=batchsize):
        mmd2, obj = val_fn(Xbatch, Ybatch)
        assert np.isfinite(mmd2)
        assert np.isfinite(obj)
        total_mmd2 += mmd2
        total_obj += obj
        n_batches += 1
    return total_mmd2 / n_batches, total_obj / n_batches


################################################################################
### Main deal

def setup(dim, criterion='mmd', biased=True, streaming_est=False, opt_log=True,
          linear_kernel=False, opt_sigma=False, init_log_sigma=0,
          net_version='basic', hotelling_reg=0,
          strat='nesterov_momentum', learning_rate=0.01, **opt_args):
    input_p = T.matrix('input_p')
    input_q = T.matrix('input_q')

    mmd2_pq, obj, rep_p, net_p, net_q, log_sigma = make_network(
        input_p, input_q, dim,
        criterion=criterion, biased=biased, streaming_est=streaming_est,
        opt_log=opt_log, linear_kernel=linear_kernel, log_sigma=init_log_sigma,
        hotelling_reg=hotelling_reg, net_version=net_version)

    params = lasagne.layers.get_all_params([net_p, net_q], trainable=True)
    if opt_sigma:
        params.append(log_sigma)
    fn = getattr(lasagne.updates, strat)
    updates = fn(obj, params, learning_rate=learning_rate, **opt_args)

    #print("Compiling...", file=sys.stderr, end='')
    train_fn = theano.function(
        [input_p, input_q], [mmd2_pq, obj], updates=updates)
    val_fn = theano.function([input_p, input_q], [mmd2_pq, obj])
    get_rep = theano.function([input_p], rep_p)
    #print("done", file=sys.stderr)

    return params, train_fn, val_fn, get_rep, log_sigma


def train(X_train, Y_train, X_val, Y_val,
          criterion='mmd', biased=True, streaming_est=False, opt_log=True,
          linear_kernel=False, hotelling_reg=0,
          init_log_sigma=0, opt_sigma=False, init_sigma_median=False,
          num_epochs=10000, batchsize=200, val_batchsize=1000,
          verbose=True, net_version='basic',
          opt_strat='nesterov_momentum', learning_rate=0.01,
          log_params=False, **opt_args):
    assert X_train.ndim == X_val.ndim == Y_train.ndim == Y_val.ndim == 2
    dim = X_train.shape[1]
    assert X_val.shape[1] == Y_train.shape[1] == Y_val.shape[1] == dim

    if verbose:
        if linear_kernel:
            print("Using linear kernel")
        elif opt_sigma:
            print("Starting with sigma = {}; optimizing it".format(
                'median' if init_sigma_median else np.exp(init_log_sigma)))
        else:
            print("Using sigma = {}".format(
                'median' if init_sigma_median else np.exp(init_log_sigma)))

    params, train_fn, val_fn, get_rep, log_sigma = setup(
            dim, criterion=criterion, linear_kernel=linear_kernel,
            biased=biased, streaming_est=streaming_est,
            hotelling_reg=hotelling_reg,
            init_log_sigma=init_log_sigma, opt_sigma=opt_sigma,
            opt_log=opt_log, net_version=net_version,
            strat=opt_strat, learning_rate=learning_rate, **opt_args)

    if log_sigma is not None and init_sigma_median:
        print("Getting median initial sigma value...", end='')
        n_samp = min(500, X_train.shape[0], Y_train.shape[0])
        samp = np.vstack([
            X_train[np.random.choice(X_train.shape[0], n_samp, replace=False)],
            Y_train[np.random.choice(Y_train.shape[0], n_samp, replace=False)],
        ])
        reps = np.vstack([
            get_rep(batch) for batch, in
            iterate_minibatches(samp, batchsize=val_batchsize)])
        D2 = euclidean_distances(reps, squared=True)
        med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
        log_sigma.set_value(make_floatX(np.log(med_sqdist / np.sqrt(2)) / 2))
        rep_dim = reps.shape[1]
        del samp, reps, D2, med_sqdist
        print("{:.3g}".format(np.exp(log_sigma.get_value())))
    else:
        rep_dim = get_rep(X_train[:1]).shape[1]

    if verbose:
        print("Input dim {}, representation dim {}".format(
            X_train.shape[1], rep_dim))
        print("Training on {} samples (batch {}), validation on {} (batch {})"
            .format(X_train.shape[0], batchsize, X_val.shape[0], val_batchsize))
        print("{} parameters to optimize: {}".format(
            len(params), ', '.join(p.name for p in params)))

    value_log = np.zeros(num_epochs + 1, dtype=[
            ('train_mmd', floatX), ('train_obj', floatX),
            ('val_mmd', floatX), ('val_obj', floatX),
            ('elapsed_time', np.float64)]
            + ([('sigma', floatX)] if opt_sigma else [])
            + ([('params', object)] if log_params else []))

    fmt = ("{: >6,}: avg train MMD^2 {: .6f} obj {: .6f},  "
           "avg val MMD^2 {: .6f}  obj {: .6f}  elapsed: {:,}s")
    if opt_sigma:
        fmt += '  sigma: {sigma:.3g}'
    def log(epoch, t_mmd2, t_obj, v_mmd2, v_job, t):
        sigma = np.exp(float(params[-1].get_value())) if opt_sigma else None
        if verbose and (epoch in {0, 5, 25, 50}
                # or (epoch < 1000 and epoch % 50 == 0)
                or epoch % 100 == 0):
            print(fmt.format(
                epoch, t_mmd2, t_obj, v_mmd2, v_obj, int(t), sigma=sigma))

        # if verbose and (epoch in {0, 5, 25, 50}
        #                 or epoch % 500 == 0):
        #     test_func(get_rep, X_val, Y_val, np.exp(log_sigma.get_value()) if log_sigma is not None else None, args)
        tup = (t_mmd2, t_obj, v_mmd2, v_obj, t)
        if opt_sigma:
            tup += (sigma,)
        if log_params:
            tup += ([p.get_value() for p in params],)
        value_log[epoch] = tup

    t_mmd2, t_obj = run_val(X_train, Y_train, batchsize, val_fn)
    v_mmd2, v_obj = run_val(X_val, Y_val, val_batchsize, val_fn)
    log(0, t_mmd2, t_obj, v_mmd2, v_obj, 0)
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        try:
            t_mmd2, t_obj = run_train_epoch(
                X_train, Y_train, batchsize, train_fn)
            v_mmd2, v_obj = run_val(X_val, Y_val, val_batchsize, val_fn)
            log(epoch, t_mmd2, t_obj, v_mmd2, v_obj, time.time() - start_time)
        except KeyboardInterrupt:
            break

    sigma = np.exp(log_sigma.get_value()) if log_sigma is not None else None
    return ([p.get_value() for p in params], [p.name for p in params],
            get_rep, value_log, sigma)


def eval_rep(get_rep, X, Y, linear_kernel=False, hotelling=False,
             sigma=None, null_samples=1000):
    Xrep = get_rep(X)
    Yrep = get_rep(Y)
    if linear_kernel:
        if hotelling:
            p_val, stat, = mmd_test.linear_hotelling_test(Xrep, Yrep)
            null_samps = np.empty(0, dtype=np.float32)
        else:
            p_val, stat, null_samps = mmd_test.linear_mmd_test(
                Xrep, Yrep, null_samples=null_samples)
    else:
        p_val, stat, null_samps, _ = mmd_test.rbf_mmd_test(
            Xrep, Yrep, bandwidth=sigma, null_samples=null_samples)
    return p_val, stat, null_samps
