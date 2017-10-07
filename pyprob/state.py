#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
from pyprob.trace import Sample, Trace
import traceback

current_trace = None

def extract_address():
    tb = traceback.extract_stack()
    # print()
    # for t in tb:
    #     print(t[0], t[1], t[2], t[3])
    frame = tb[-4]
    # return '{0}/{1}/{2}'.format(frame[1], frame[2], frame[3])
    return '{0}/{1}'.format(frame[1], frame[2])

def sample(distribution):
    global current_trace
    value = distribution.sample()
    if current_trace is not None:
        address = extract_address()
        sample = Sample(address, distribution, value)
        current_trace.add_sample(sample)
        print('Added sample {}'.format(sample))

    return value

def observe(value, observed_value):
    print('Observed {}'.format(observed_value))
    return
