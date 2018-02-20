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
        self.lstm_input = None
        self.lstm_output = None

    def __repr__(self):
        return 'Sample(address_suffixed:{}, distribution:{}, value:{})'.format(
            self.address_suffixed,
            str(self.distribution),
            self.value.data.cpu().numpy().tolist()
        )

    def cuda(self, device=None):
        if self.value is not None:
            self.value = self.value.cuda(device)
        # self.distribution.cuda(device)

    def cpu(self):
        if self.value is not None:
            self.value = self.value.cpu()
        # self.distribution.cpu()


class Trace(object):
    def __init__(self):
        self.observes = []
        self.observes_log_prob = []
        self.observes_variable = None
        self.observes_embedding = None
        self.samples = []
        self.samples_log_prob = []
        self.length = 0
        self.result = None
        self.log_prob = 0

    def __repr__(self):
        return 'Trace(length:{}, samples:[{}], observes_variable:{}, result:{}, log_prob:{})'.format(
            self.length,
            ', '.join([str(sample) for sample in self.samples]),
            self.observes_variable.data.cpu().numpy().tolist(),
            self.result.data.cpu().numpy().tolist(),
            float(self.log_prob)
        )

    def _find_last_sample(self, address):
        indices = [i for i, sample in enumerate(self.samples) if sample.address == address]
        if len(indices) == 0:
            return None
        else:
            return max(indices)

    def addresses(self):
        return '; '.join([sample.address for sample in self.samples])

    def addresses_suffixed(self):
        return '; '.join([sample.address_suffixed for sample in self.samples])

    def compute_log_prob(self):
        self.log_prob = sum(self.samples_log_prob) + sum(self.observes_log_prob)

    def set_result(self, r):
        self.result = r

    def add_sample(self, sample, log_prob=0, replace=False):
        if replace:
            i = self._find_last_sample(sample.address)
            if i is not None:
                self.samples[i] = sample
                self.samples_log_prob[i] = log_prob
                return
        self.samples.append(sample)
        self.samples_log_prob.append(log_prob)
        self.length += 1

    def add_observe(self, observe, log_prob=0):
        self.observes.append(observe)
        self.observes_log_prob.append(log_prob)

    def pack_observes_to_variable(self):
        self.observes_variable = util.pack_observes_to_variable(self.observes)

    def cuda(self, device=None):
        if self.observes_variable is not None:
            self.observes_variable = self.observes_variable.cuda(device)
        if self.observes_embedding is not None:
            self.observes_embedding = self.observes_embedding.cuda(device)
        for sample in self.samples:
            sample.cuda(device)

    def cpu(self):
        if self.observes_variable is not None:
            self.observes_variable = self.observes_variable.cpu()
        if self.observes_embedding is not None:
            self.observes_embedding = self.observes_embedding.cpu()
        for sample in self.samples:
            sample.cpu()
