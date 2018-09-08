import torch


class Variable():
    def __init__(self, distribution=None, value=None, address_base=None, address=None, instance=None, log_prob=None, control=None, replace=None, observable_name=None, observed=False, reused=False):
        self.distribution = distribution
        self.value = value
        self.address_base = address_base
        self.address = address
        self.instance = instance
        self.log_prob = log_prob
        self.control = control
        self.replace = replace
        self.observable_name = observable_name
        self.observable = observable_name is not None
        self.observed = observed
        self.reused = reused

    def __repr__(self):
        return 'Variable(control:{}, replace:{}, observable_name:{}, observed:{}, address:{}, distribution:{}, value:{})'.format(
            self.control,
            self.replace,
            self.observable_name,
            self.observed,
            self.address,
            str(self.distribution),
            str(self.value))


class Trace():
    def __init__(self):
        self.variables = []
        self.variables_controlled = []
        self.variables_uncontrolled = []
        self.variables_replaced = []
        self.variables_observed = []
        self.variables_observable = []
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0
        self.length_controlled = 0

    def __repr__(self):
        return 'Trace(all:{:,}, controlled:{:,}, replaced:{:,}, observeable:{:,}, observed:{:,}, uncontrolled:{:,}, log_prob:{}, log_importance_weight:{})'.format(
            len(self.variables),
            len(self.variables_controlled),
            len(self.variables_replaced),
            len(self.variables_observable),
            len(self.variables_observed),
            len(self.variables_uncontrolled),
            str(self.log_prob),
            str(self.log_importance_weight))

    def add(self, variable):
        self.variables.append(variable)
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable

    def end(self, result):
        self.result = result
        replaced_indices = []
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if variable.control and i not in replaced_indices:
                if variable.replace:
                    for j in range(i + 1, len(self.variables)):
                        if self.variables[j].address_base == variable.address_base:
                            self.variables_replaced.append(variable)
                            variable = self.variables[j]
                            replaced_indices.append(j)
                self.variables_controlled.append(variable)
        self.variables_uncontrolled = [v for v in self.variables if (not v.control) and (not v.observed)]
        self.variables_observed = [v for v in self.variables if v.observed]
        self.variables_observable = [v for v in self.variables if v.observable]
        self.log_prob = sum([torch.sum(v.log_prob) for v in self.variables if v.control or v.observed])
        self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self.variables_observed])
        self.length = len(self.variables)
        self.length_controlled = len(self.variables_controlled)

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0
