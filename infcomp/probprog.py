class Sample(object):
    def __init__(self):
        self.address = None
        self.instance = None
        self.value = None
        self.value_dim = None
        self.proposal = None
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
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.probabilities = None
    def __repr__(self):
        return 'UniformDiscreteProposal(min:{0}; max:{1}; probabilities:{2})'.format(self.min, self.max, self.probabilities)
    __str__ = __repr__
    def set_proposalparams(self, p):
        self.probabilities = p
    def name(self):
        return 'UniformDiscreteProposal'

class NormalProposal(object):
    def __init__(self):
        self.mean = None
        self.std = None
    def __repr__(self):
        return 'NormalProposal(mean:{0}; std:{1})'.format(self.mean, self.std)
    __str__ = __repr__
    def set_proposalparams(self, tensor_of_mean_std):
        self.mean = tensor_of_mean_std[0]
        self.std = tensor_of_mean_std[1]
    def name(self):
        return 'NormalProposal'
