#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util


class Sample(object):
    def __init__(self):
        self.address = None
        self.address_suffixed = None
        self.instance = None
        self.value = util.Tensor()
        self.value_dim = None
        self.distribution = None
        self.lstm_input = None
        self.lstm_output = None
    def __repr__(self):
        return 'Sample({0}, {1}, {2}, {3}, {4})'.format(self.address, self.address_suffixed, self.instance, self.value.size(), str(self.distribution))
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
        self.length = None
    def __repr__(self):
        return 'Trace(length:{0}, samples:[{1}], observes_tensor.dim():{2})'.format(self.length, ', '.join([str(sample) for sample in self.samples]), self.observes_tensor.dim())
    __str__ = __repr__
    def addresses(self):
        return '; '.join([sample.address for sample in self.samples])
    def addresses_suffixed(self):
        return '; '.join([sample.address_suffixed for sample in self.samples])
    def add_observe(self, o):
        self.observes.append(o)
        
    def set_observes(self, o):
        self.observes_tensor = o
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
