import torch

from . import util


class Variable():
    def __init__(self, distribution=None, value=None, address_base=None, address=None, instance=None, log_prob=None, log_importance_weight=None, control=False, replace=False, name=None, observed=False, reused=False, tagged=False):
        self.distribution = distribution
        if value is None:
            self.value = None
        else:
            self.value = util.to_tensor(value)
        self.address_base = address_base
        self.address = address
        self.instance = instance
        if log_prob is None:
            self.log_prob = None
        else:
            self.log_prob = util.to_tensor(log_prob)
        if log_importance_weight is None:
            self.log_importance_weight = None
        else:
            self.log_importance_weight = float(log_importance_weight)
        self.control = control
        self.replace = replace
        self.name = name
        self.observable = ((not tagged) and (name is not None)) or observed
        self.observed = observed
        self.reused = reused
        self.tagged = tagged

    def __repr__(self):
        return 'Variable(name:{}, control:{}, replace:{}, observable:{}, observed:{}, tagged:{}, address:{}, distribution:{}, value:{}: log_prob:{})'.format(
            self.name,
            self.control,
            self.replace,
            self.observable,
            self.observed,
            self.tagged,
            self.address,
            str(self.distribution),
            str(self.value),
            str(self.log_prob))

    def to(self, device):
        if self.value is not None:
            self.value.to(device=device)
        # if self.distribution is not None:
        #     self.distribution.to(device=device)

    def __hash__(self):
        return hash(self.address + str(self.value) + str(self.control) + str(self.replace) + str(self.observed) + str(self.tagged))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Trace():
    def __init__(self):
        self.variables = []
        self.variables_controlled = []
        self.variables_uncontrolled = []
        self.variables_replaced = []
        self.variables_observed = []
        self.variables_observable = []
        self.variables_tagged = []
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.named_variables = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0
        self.length_controlled = 0
        self.execution_time_sec = None

    def __repr__(self):
        return 'Trace(all:{:,}, controlled:{:,}, replaced:{:,}, observeable:{:,}, observed:{:,}, tagged:{:,}, uncontrolled:{:,}, log_prob:{}, log_importance_weight:{})'.format(
            len(self.variables),
            len(self.variables_controlled),
            len(self.variables_replaced),
            len(self.variables_observable),
            len(self.variables_observed),
            len(self.variables_tagged),
            len(self.variables_uncontrolled),
            str(self.log_prob),
            str(self.log_importance_weight))

    def add(self, variable):
        self.variables.append(variable)
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        replaced_indices = []
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if variable.name is not None:
                self.named_variables[variable.name] = variable
            if variable.control and i not in replaced_indices:
                if variable.replace:
                    for j in range(i + 1, len(self.variables)):
                        if self.variables[j].address_base == variable.address_base:
                            self.variables_replaced.append(variable)
                            variable = self.variables[j]
                            replaced_indices.append(j)
                self.variables_controlled.append(variable)
        self.variables_uncontrolled = [v for v in self.variables if (not v.control) and (not v.observed) and (not v.tagged)]
        self.variables_observed = [v for v in self.variables if v.observed]
        self.variables_observable = [v for v in self.variables if v.observable]
        self.variables_tagged = [v for v in self.variables if v.tagged]
        self.log_prob = sum([torch.sum(v.log_prob) for v in self.variables if v.control or v.observed])
        self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self.variables_observed])
        self.length = len(self.variables)
        self.length_controlled = len(self.variables_controlled)
        replaced_log_importance_weights = {}
        for variable in self.variables:
            if variable.log_importance_weight is not None:
                if variable.replace:
                    replaced_log_importance_weights[variable.address_base] = variable.log_importance_weight
                else:
                    self.log_importance_weight += variable.log_importance_weight
        for _, log_importance_weight in replaced_log_importance_weights.items():
            self.log_importance_weight += log_importance_weight

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def to(self, device):
        for variable in self.variables:
            variable.to(device)

    def __hash__(self):
        h = [hash(variable) for variable in self.variables]
        return hash(sum(h))

    def __eq__(self, other):
        return hash(self) == hash(other)
