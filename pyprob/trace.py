from . import util


class Sample(object):
    def __init__(self, address, distribution, value):
        self.address = address
        self.distribution = distribution
        if distribution is None:
            self.address_suffixed = address
        else:
            self.address_suffixed = address + distribution.address_suffix
        self.value = util.to_variable(value)


class Trace(object):
    def __init__(self):
        self.observes = []
        self.samples = []
        self.length = 0
        self.result = None
        self.log_prob = 0

    def add_sample(self, s):
        self.samples.append(s)
        self.length += 1

    def add_observe(self, o):
        self.observes.append(o)

    def add_log_prob(self, p):
        self.log_prob += p

    def set_result(self, r):
        self.result = r
