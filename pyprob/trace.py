import torch
import hashlib

from . import util

class Variable():
    def __init__(self, distribution=None, value=None, address_base=None,
                 address=None, instance=None, log_prob=None,
                 log_importance_weight=None, control=False, constants={},
                 name=None, observed=False, reused=False, tagged=False,
                 replace=False, distribution_name=None, distribution_args=None):
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
        self.name = name
        self.observed = observed
        self.reused = reused
        self.tagged = tagged
        self.constants = constants
        self.replace = replace
        self.distribution=distribution
        if distribution:
            self.distribution_name = distribution.name
            self.distribution_args = distribution.args
        else:
            self.distribution_name = distribution_name
            self.distribution_args = distribution_args

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned variables in offline training datasets
        return 'Variable(name:{}, control:{}, constans:{}, observed:{}, tagged:{}, address:{}, distribution_name:{}, value:{}: log_prob:{})'.format(
            self.name if hasattr(self, 'name') else 'Unknown',
            self.control if hasattr(self, 'control') else 'Unknown',
            self.constants if hasattr(self, 'constants') else 'Unknown',
            self.observed if hasattr(self, 'observed') else 'Unknown',
            self.tagged if hasattr(self, 'tagged') else 'Unknown',
            self.address if hasattr(self, 'address') else 'Unknown',
            str(self.distribution_name) if hasattr(self, 'distribution_name') else 'Unknown',
            str(self.value) if hasattr(self, 'value') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown')

    def to(self, device):
        if self.value is not None:
            self.value.to(device=device)

    def __hash__(self):
        return hash(self.address + str(self.value) + str(self.control) + str(self.observed) + str(self.tagged))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Trace():
    def __init__(self, trace_hash=None):
        self.variables = []
        self.variables_observed = {}
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.result = None
        self.log_prob = 0.
        self.log_prob_observed = 0.
        self.log_importance_weight = 0.
        self.length = 0
        self.execution_time_sec = None
        self.trace_hash = trace_hash

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned traces in offline training datasets
        return 'Trace(number of variables:{:,}, observed:{}, log_prob:{}, log_importance_weight:{})'.format(
            self.length,
            '{:,}'.format(len(self.variables_observed)) if hasattr(self, 'variables_observed') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown',
            str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown')

    def add(self, variable):
        self.variables.append(variable)
        if variable.observed:
            # HAS TO HAVE A NAME
            self.variables_observed[variable.name] = variable
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable
        self.log_prob += torch.sum(variable.log_prob)
        if variable.log_importance_weight is not None:
            self.log_importance_weight += variable.log_importance_weight

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        self.length = len(self.variables)

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def to(self, device):
        for variable in self.variables:
            variable.to(device)

    def hash(self):
        address_list = [variable.address for variable in self.variables]
        trace_hash = hashlib.sha224(''.join(address_list).encode()).hexdigest()
        return trace_hash

    def __eq__(self, other):
        return self.hash() == other.hash()
