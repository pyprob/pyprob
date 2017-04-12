class Sample(object):
    def __init__(self, address, instance, value, proposal):
        self.address = address
        self.instance = instance
        self.value = value
        self.value_dim = value.nelement()
        self.proposal = proposal
    def __repr__(self):
        return 'Sample({0}, {1}, {2}, {3})'.format(self.address, self.instance, self.value.size(), self.proposal)
    __str__ = __repr__

class Trace(object):
    def __init__(self):
        self.observes = None
        self.samples = []
        self.length = None
        self.hash = None
    def __repr__(self):
        return 'Trace(length:{0}; samples:{1}; observes:{2}'.format(self.length, '|'.join(['{0}({1})'.format(sample.address, sample.instance) for sample in  self.samples]), self.observes.size()) + ')'
    __str__ = __repr__
    def set_observes(self, o):
        self.observes = o
    def add_sample(self, s):
        self.samples.append(s)
        self.length = len(self.samples)

class UniformDiscreteProposal(object):
    def __init__(self, min, max, probabilities=None):
        self.min = min
        self.max = max
        self.probabilities = probabilities
    def __repr__(self):
        return 'UniformDiscreteProposal(min:{0}; max:{1}; probabilities:{2})'.format(self.min, self.max, self.probabilities)
    __str__ = __repr__
    def name(self):
        return 'UniformDiscreteProposal'
