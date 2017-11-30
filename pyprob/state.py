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
trace_mode = 'inference'
artifact = None

def set_mode(mode):
    global trace_mode
    if mode == 'inference':
        trace_mode = 'inference'
    elif mode == 'compilation':
        trace_mode = 'compilation'
    elif mode == 'compiled_inference':
        trace_mode = 'compiled_inference'
    else:
        raise Exception('Unknown mode: {}. Use one of (inference, compilation, compiled_inference).'.format(mode))

def set_artifact(art):
    global artifact
    artifact = art

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
        current_sample = Sample(address, distribution, value)
        if trace_mode == 'compiled_inference':
            previous_sample = None
            if current_trace.length > 0:
                previous_sample = current_trace.samples[-1]
            proposal_distribution = artifact.forward(previous_sample, current_sample, volatile=True)
            proposal_distribution = pyprob.distributions.Normal(proposal_distribution.proposal_mean, proposal_distribution.proposal_std)
            value = proposal_distribution.sample()
            current_sample = Sample(address, distribution, value)
            current_trace.add_log_p(distribution.log_pdf(value) - proposal_distribution.log_pdf(value))
        current_trace.add_sample(current_sample)
        # current_trace.add_log_p(distribution.log_pdf(value))
        # print('Added sample {}'.format(sample))
    return value

def observe(distribution, value):
    global current_trace
    if current_trace is not None:
        if trace_mode == 'compilation':
            current_trace.add_observe(distribution.sample())
        else:
            current_trace.add_observe(value)
        current_trace.add_log_p(distribution.log_pdf(value))
        # print('Added observe {}'.format(value))
    return
