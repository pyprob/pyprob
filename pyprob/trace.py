import torch
import pickle
import copy
from collections import Counter

from . import util


class Variable():
    def __init__(self, distribution=None, value=None, address_base=None, address=None, instance=None, log_prob=None, log_importance_weight=None, control=False, name=None, observed=False, reused=False, tagged=False):
        self.distribution = distribution
        # if value is None:
        #     self.value = None
        # else:
        #     self.value = util.to_tensor(value)
        self.value = value
        self.address_base = address_base
        self.address = address
        self.instance = instance
        self.log_prob = util.to_tensor(log_prob)
        if log_importance_weight is None:
            self.log_importance_weight = None
        else:
            self.log_importance_weight = float(log_importance_weight)
        self.control = control
        self.name = name
        self.observable = ((not tagged) and (name is not None)) or observed
        self.observed = observed
        self.reused = reused
        self.tagged = tagged

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned variables in offline training datasets
        return 'Variable(name:{}, observable:{}, observed:{}, tagged:{}, control:{}, address:{}, distribution:{}, value:{}, log_importance_weight:{}, log_prob:{})'.format(
            self.name if hasattr(self, 'name') else 'Unknown',
            self.observable if hasattr(self, 'observable') else 'Unknown',
            self.observed if hasattr(self, 'observed') else 'Unknown',
            self.tagged if hasattr(self, 'tagged') else 'Unknown',
            self.control if hasattr(self, 'control') else 'Unknown',
            self.address if hasattr(self, 'address') else 'Unknown',
            str(self.distribution) if hasattr(self, 'distribution') else 'Unknown',
            str(self.value) if hasattr(self, 'value') else 'Unknown',
            str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown')

    def clone(self):
        return copy.deepcopy(self)

    def to(self, device):
        # The 'hasattr' checks below are for handling pruned variables in offline training datasets
        ret = self.clone()
        if hasattr(ret, 'value'):
            if (ret.value is not None) and torch.is_tensor(ret.value):
                ret.value = ret.value.to(device=device)
        if hasattr(ret, 'log_prob'):
            if (ret.log_prob is not None) and torch.is_tensor(ret.log_prob):
                ret.log_prob = ret.log_prob.to(device=device)
        if hasattr(ret, 'distribution'):
            if ret.distribution is not None:
                ret.distribution = ret.distribution.to(device=device)
        return ret

    def __hash__(self):
        return hash(self.address + str(self.value) + str(self.control) + str(self.observed) + str(self.tagged))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Trace():
    def __init__(self):
        self.variables = []
        self.variables_controlled = []
        self.variables_uncontrolled = []
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
        # The 'Unknown' cases below are for handling pruned traces in offline training datasets
        return 'Trace(variables:{:,}, observable:{}, observed:{}, tagged:{}, controlled:{:,}, uncontrolled:{}, log_prob:{}, log_importance_weight:{})'.format(
            self.length,
            '{:,}'.format(len(self.variables_observable)) if hasattr(self, 'variables_observable') else 'Unknown',
            '{:,}'.format(len(self.variables_observed)) if hasattr(self, 'variables_observed') else 'Unknown',
            '{:,}'.format(len(self.variables_tagged)) if hasattr(self, 'variables_tagged') else 'Unknown',
            self.length_controlled,
            '{:,}'.format(len(self.variables_uncontrolled)) if hasattr(self, 'variables_uncontrolled') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown',
            str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown')

    def add(self, variable):
        self.variables.append(variable)
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        for i in range(len(self.variables)):
            variable = self.variables[i]
            if variable.name is not None:
                self.named_variables[variable.name] = variable
            if variable.control:
                self.variables_controlled.append(variable)
        self.variables_uncontrolled = [v for v in self.variables if (not v.control) and (not v.observed) and (not v.tagged)]
        self.variables_observed = [v for v in self.variables if v.observed]
        self.variables_observable = [v for v in self.variables if v.observable]
        self.variables_tagged = [v for v in self.variables if v.tagged]
        self.log_prob = sum([torch.sum(v.log_prob) for v in self.variables if v.control or v.observed])
        self.log_prob_observed = sum([torch.sum(v.log_prob) for v in self.variables_observed])
        self.length = len(self.variables)
        self.length_controlled = len(self.variables_controlled)
        for variable in self.variables:
            if variable.log_importance_weight is not None:
                self.log_importance_weight += variable.log_importance_weight

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def address_counts(self, use_address_base=True):
        if use_address_base:
            addresses = [v.address_base for v in self.variables]
        else:
            addresses = [v.address for v in self.variables]
        return Counter(addresses)

    def clone(self):
        return copy.deepcopy(self)

    def to(self, device):
        # The 'hasattr' checks below are for handling pruned variables in offline training datasets
        ret = self.clone()
        if hasattr(ret, 'variables'):
            for i, variable in enumerate(ret.variables):
                ret.variables[i] = variable.to(device)
        if hasattr(ret, 'variables_controlled'):
            for i, variable in enumerate(ret.variables_controlled):
                ret.variables_controlled[i] = variable.to(device)
        if hasattr(ret, 'variables_uncontrolled'):
            for i, variable in enumerate(ret.variables_uncontrolled):
                ret.variables_uncontrolled[i] = variable.to(device)
        if hasattr(ret, 'variables_observed'):
            for i, variable in enumerate(ret.variables_observed):
                ret.variables_observed[i] = variable.to(device)
        if hasattr(ret, 'variables_observable'):
            for i, variable in enumerate(ret.variables_observable):
                ret.variables_observable[i] = variable.to(device)
        if hasattr(ret, 'variables_tagged'):
            for i, variable in enumerate(ret.variables_tagged):
                ret.variables_tagged[i] = variable.to(device)

        if hasattr(ret, 'variables_dict_address'):
            for key, variable in ret.variables_dict_address.items():
                ret.variables_dict_address[key] = variable.to(device)
        if hasattr(ret, 'variables_dict_address_base'):
            for key, variable in ret.variables_dict_address_base.items():
                ret.variables_dict_address_base[key] = variable.to(device)
        if hasattr(ret, 'named_variables'):
            for key, variable in ret.named_variables.items():
                ret.named_variables[key] = variable.to(device)

        return ret

    def variable_sizes(self):
        vars_sorted = sorted(self.variables, key=lambda v: len(pickle.dumps(v)), reverse=True)
        vars_sorted_sizes = list(map(lambda v: len(pickle.dumps(v)), vars_sorted))
        return vars_sorted, vars_sorted_sizes

    def __len__(self):
        return self.length

    def __hash__(self):
        h = [hash(variable) for variable in self.variables]
        return hash(sum(h))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __getitem__(self, variable_name):
        if variable_name in self.named_variables:
            return self.named_variables[variable_name].value
        else:
            raise RuntimeError('Trace does not include variable with name: {}'.format(variable_name))

    def __contains__(self, variable_name):
        return variable_name in self.named_variables
