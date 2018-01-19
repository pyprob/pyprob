#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
import torch

class Sample(object):

    def __init__(self, address, distribution, value):
        self.address = address
        self.distribution = distribution
        if distribution is None:
            self.address_suffixed = address
        else:
            self.address_suffixed = address + distribution.address_suffix
        # self.instance = None
        self.value = util.to_tensor(value)
        self.value_dim = None
        self.lstm_input = None
        self.lstm_output = None

    def __repr__(self):
        return 'Sample(address_suffixed:{0}, distribution:{1}, value:{2})'.format(
            self.address_suffixed,
            str(self.distribution),
            self.value.cpu().numpy().tolist()
        )
    __str__ = __repr__

    def cuda(self, device_id=None):
        if not self.value is None:
            self.value = self.value.cuda(device_id)
        self.distribution.cuda(device_id)

    def cpu(self):
        if not self.value is None:
            self.value = self.value.cpu()
        self.distribution.cpu()

class Trace(object):
    def __init__(self):
        self.observes = []
        self.observes_tensor = None
        self.observes_embedding = None
        self.samples = []
        self.length = 0
        self.result = None
        self.log_p = 0

    def __repr__(self):
        return 'Trace(length:{0}, samples:[{1}], observes_tensor:{2}, result:{3}, log_p:{4})'.format(
            self.length,
            ', '.join([str(sample) for sample in self.samples]),
            self.observes_tensor.cpu().numpy().tolist(),
            str(self.result),
            self.log_p
        )
    __str__ = __repr__

    def addresses(self):
        return '; '.join([sample.address for sample in self.samples])

    def addresses_suffixed(self):
        return '; '.join([sample.address_suffixed for sample in self.samples])

    def add_log_p(self, p):
        self.log_p += p

    def set_result(self, r):
        self.result = r

    def add_observe(self, o):
        self.observes.append(o)

    def set_observes_tensor(self, o):
        self.observes_tensor = o

    def pack_observes_to_tensor(self):
        self.observes_tensor = util.pack_observes_to_tensor(self.observes)

    def add_sample(self, s):
        self.samples.append(s)
        self.length = len(self.samples)

    def cuda(self, device_id=None):
        if not self.observes_tensor is None:
            self.observes_tensor = self.observes_tensor.cuda(device_id)
        for i in range(len(self.samples)):
            self.samples[i].cuda(device_id)

    def cpu(self):
        if not self.observes_tensor is None:
            self.observes_tensor = self.observes_tensor.cpu()
        for i in range(len(self.samples)):
            self.samples[i].cpu()
