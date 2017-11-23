#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
from pyprob.trace import Sample, Trace
import traceback
import torch

current_trace = None

def begin_trace():
    global current_trace
    current_trace = Trace()

def end_trace():
    global current_trace
    current_trace.pack_observes_to_tensor()
    ret = current_trace
    current_trace = None
    return ret

def extract_address():
    tb = traceback.extract_stack()
    # print()
    # for t in tb:
    #     print(t[0], t[1], t[2], t[3])
    frame = tb[-3]
    # return '{0}/{1}/{2}'.format(frame[1], frame[2], frame[3])
    return '{0}/{1}'.format(frame[1], frame[2])

def sample(distribution):
    global current_trace
    value = distribution.sample()
    if current_trace is not None:
        address = extract_address()
        sample = Sample(address, distribution, value)
        current_trace.add_sample(sample)
        # current_trace.add_log_p(distribution.log_pdf(value))
        # print('Added sample {}'.format(sample))

    return value

def observe(distribution, value):
    if current_trace is not None:
        current_trace.add_observe(value)
        current_trace.add_log_p(distribution.log_pdf(value))
        # print('Added observe {}'.format(value))
    return
