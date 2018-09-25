#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  00:23
Date created:  25/09/2018

License: MIT
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:28
Date created:  21/09/2018

License: MIT
'''
import time
import torch
import numpy as np
import scipy.stats as ss
from torch.autograd import Variable
from Distributions import *
from HMC import HMC
import sys
PATH =  '/data/greyplover/not-backed-up/aims/aims16/bgramhan/results/hmc'
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

### model param
a = np.exp(-5)
dim =100

########### MODEL
def f(x0, compute_grad=True):
    '''
    :param x0: dim by 1
    :return:
    '''
    x = Variable(x0.data, requires_grad = True)

    if torch.max(torch.pow(x, 2)).data[0] <= 9 :
        logp = -(torch.sum(torch.pow(x,2)) * a)
        logp = logp.unsqueeze(1)
    elif 9 < torch.max(torch.pow(x, 2)).data[0] <= 36 :
        logp = -((torch.sum(torch.pow(x, 2)) * a) + np.log(10))
        logp = logp.unsqueeze(1)
    else:
	logp = Variable(torch.Tensor([[-np.inf]]))
        return logp, Variable(torch.Tensor([[0]])), logp

    if compute_grad:
        grad = torch.autograd.grad(logp, x)[0]  #grad var
    else:
	grad = 0

    aux = logp
    return logp, grad, aux

def f_update(x, dx, j, aux):
    logp_prev = aux

    x_new = x.clone()
    x_new.data[j] = x_new.data[j] + dx
    logp ,_,_ = f(x_new, False)
    logp_diff = logp - logp_prev
    aux_new = logp

    return logp_diff, aux_new

# DHMC
def run_hmc_parallel(chain_id,PATH):
    print('Running chain {}'.format(chain_id))
    n_param = dim
    n_disc = 0
    hmc_obj = HMC(f, n_disc, n_param, f_update)
    inference= 'hmc'
    dt = np.array([.1, 0.15])
    nstep = [5, 10]
    n_burnin = 0
    n_sample = 100000000

    seed = chain_id
    torch.manual_seed(seed=seed)
    np.random.seed(seed)

    print("Start Sampling Chain = ", chain_id, "\n")
    x0 = Variable(torch.zeros(dim,1))
    x0.data[:,0] = torch.rand(dim) * 12 - 6
    t0 = time.time()
    samples, accept =\
        hmc_obj.run_hmc(x0, dt, nstep, n_burnin, n_sample, seed=seed)
    t1 = time.time()
    total = t1-t0
    print('Total : {0}'.format(total))
    print(' Total time for {3} to generate {0} samples in {1} dims is: {2}'.format(n_sample, dim, total, inference))
    df = pd.DataFrame(samples.numpy())
    PATH = PATH +'/data/{}d/'.format(dim)
    os.makedirs(PATH, exist_ok=True)
    df.to_csv(PATH + 'hmc_' + str(n_sample) + '_chain_' + str(chain_id) +'.csv')

chain_id = range(0,10)
results = Parallel(n_jobs=5)(delayed(run_hmc_parallel)(i,PATH) for i in chain_id)






