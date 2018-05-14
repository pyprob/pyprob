from . import util, TrainingObservation


class Sample(object):
    def __init__(self, distribution, value, address_base, address, instance, log_prob=None, control=False, replace=False, observed=False, reused=False):
        self.address_base = address_base
        self.address = address
        self.distribution = distribution
        self.instance = instance
        self.value = util.to_variable(value)
        self.control = control
        self.replace = replace
        self.observed = observed
        self.reused = reused
        if log_prob is None:
            self.log_prob = distribution.log_prob(value)
        else:
            self.log_prob = util.to_variable(log_prob)
        self.lstm_input = None
        self.lstm_output = None

    def __repr__(self):
        return 'Sample(control:{}, replace:{}, observed:{}, address:{}, distribution:{}, value:{})'.format(
            self.control,
            self.replace,
            self.observed,
            self.address,
            str(self.distribution),
            str(self.value)
        )

    def cuda(self, device=None):
        if self.value is not None:
            self.value = self.value.cuda(device)
        # self.distribution.cuda(device)
        return self

    def cpu(self):
        if self.value is not None:
            self.value = self.value.cpu()
        # self.distribution.cpu()
        return self


class Trace(object):
    def __init__(self):
        self.samples = []  # controlled
        self.samples_uncontrolled = []
        self.samples_replaced = []
        self.samples_observed = []
        self._samples_all = []
        self._samples_all_dict_address = {}
        self._samples_all_dict_address_base = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0

    def __repr__(self):
        return 'Trace(all:{:,}, controlled:{:,}, replaced:{:,}, uncontrolled:{:,}, observed:{:,}, log_prob:{:,.2f}, log_importance_weight:{:,.2f})'.format(len(self._samples_all), len(self.samples), len(self.samples_replaced), len(self.samples_uncontrolled), len(self.samples_observed), float(self.log_prob), float(self.log_importance_weight))

    def addresses(self):
        return '; '.join([sample.address for sample in self.samples])

    def end(self, result):
        self.result = result
        self.samples = []
        self.samples_replaced = []
        replaced_indices = []
        for i in range(len(self._samples_all)):
            sample = self._samples_all[i]
            if sample.control and i not in replaced_indices:
                if sample.replace:
                    for j in range(i + 1, len(self._samples_all)):
                        if self._samples_all[j].address_base == sample.address_base:
                            self.samples_replaced.append(sample)
                            sample = self._samples_all[j]
                            replaced_indices.append(j)
                self.samples.append(sample)
        self.samples_uncontrolled = [s for s in self._samples_all if (not s.control) and (not s.observed)]
        self.samples_observed = [s for s in self._samples_all if s.observed]
        self.log_prob_observed = util.to_variable(sum([util.safe_torch_sum(s.log_prob) for s in self.samples_observed])).view(-1)

        self.log_prob = util.to_variable(sum([util.safe_torch_sum(s.log_prob) for s in self._samples_all if s.control or s.observed])).view(-1)
        self.length = len(self.samples)

    def pack_observes(self, training_observation=TrainingObservation.OBSERVE_DIST_SAMPLE):
        if training_observation == TrainingObservation.OBSERVE_DIST_SAMPLE:
            return util.pack_observes_to_variable([s.distribution.sample()[0] for s in self.samples_observed])
        else:  # training_observation == TrainingObservation.OBSERVE_DIST_MEAN
            return util.pack_observes_to_variable([s.distribution.mean[0] for s in self.samples_observed])

    def last_instance(self, address_base):
        if address_base in self._samples_all_dict_address_base:
            return self._samples_all_dict_address_base[address_base].instance
        else:
            return 0

    def add_sample(self, sample):
        self._samples_all.append(sample)
        self._samples_all_dict_address[sample.address] = sample
        self._samples_all_dict_address_base[sample.address_base] = sample

    def cuda(self, device=None):
        for sample in self._samples_all:
            sample.cuda(device)

    def cpu(self):
        for sample in self._samples_all:
            sample.cpu()
