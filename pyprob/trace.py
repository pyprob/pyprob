import torch
import hashlib

from . import util

class Variable():
    def __init__(self, distribution=None, value=None, address_base=None,
                 address=None, instance=None, log_prob=None,
                 log_importance_weight=None, control=False, constants={},
                 name=None, observed=False, reused=False, tagged=False,
                 replace=False, distribution_name=None, distribution_args=None,
                 accepted=True):
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
        self.distribution = distribution
        if distribution:
            self.distribution_name = distribution.name
            self.distribution_args = distribution.get_input_parameters()
        else:
            self.distribution_name = distribution_name
            self.distribution_args = distribution_args
        self.accepted = accepted

    def __repr__(self):
        # The 'Unknown' cases below are for handling pruned variables in offline training datasets
        return 'Variable(name:{}, control:{}, constants:{}, observed:{}, tagged:{}, replace:{}, reused:{}, address:{}, distribution_name:{}, value:{}: log_prob:{}, log_importance_weight:{})'.format(
            self.name if hasattr(self, 'name') else 'Unknown',
            self.control if hasattr(self, 'control') else 'Unknown',
            self.constants if hasattr(self, 'constants') else 'Unknown',
            self.observed if hasattr(self, 'observed') else 'Unknown',
            self.tagged if hasattr(self, 'tagged') else 'Unknown',
            self.replace if hasattr(self, 'replace') else 'Unknown',
            self.reused if hasattr(self, 'reused') else 'Unknown',
            self.address if hasattr(self, 'address') else 'Unknown',
            str(self.distribution_name) if hasattr(self, 'distribution_name') else 'Unknown',
            str(self.value) if hasattr(self, 'value') else 'Unknown',
            str(self.log_prob) if hasattr(self, 'log_prob') else 'Unknown',
            str(self.log_importance_weight) if hasattr(self, 'log_importance_weight') else 'Unknown')

    def to(self, device):
        if self.value is not None:
            self.value.to(device=device)

    def __hash__(self):
        # WHAT ABOUT HERE?
        return hash(self.address + str(self.value) + str(self.control) + str(self.observed) + str(self.tagged))

    def __eq__(self, other):
        return hash(self) == hash(other)


class RejectionSamplingStack:
    def __init__(self):
        '''
        Stores a stack of tuples (rejection sampling entry, previous variable, (LSTM hidden state, network's previous variable))
        if a network is present and has a hidden state, otherwise,
        (rejection sampling entry, previous variable, (none, network's previous variable))
        where network's previous variable is the variable before entering the rejection sampling loop for the first time.
        (LSTM hidden state, network's previous variable) are used to restore network's state after retrying.
        '''
        self._stack = []

    def push(self, entry, previous_variable, network_state):
        # network_state should be a tuple of the format (LSTM hidden state or None, network's previous variable)
        if len(network_state) != 2:
            raise ValueError("Network state is expected to be a tuple of size two with the format (LSTM hidden state or None, network's previous variable)")
        self._stack.append([entry, previous_variable, network_state])

    def updateTop(self, entry, previous_variable):
        self._stack[-1][0] = entry
        self._stack[-1][1] = previous_variable

    def pop(self):
        self._stack.pop()

    @property
    def top_entry(self):
        return self._stack[-1][0]

    @property
    def top_previous_variable(self):
        return self._stack[-1][1]

    @property
    def top_network_state(self):
        return self._stack[-1][2]

    def size(self):
        return len(self._stack)

    def isempty(self):
        return self.size() == 0


class RSEntry:
    def __init__(self, address=None, address_base=None, name=None, instance=None, control=None, iteration=None):
        self.address = address
        self.address_base = address_base
        self.name = name
        self.instance = instance
        self.control = control
        self.iteration=iteration
        self.log_importance_weight = None
        self.log_prob = None
        

    def __repr__(self):
        return f'RSEntry(name:{self.name}, address:{self.address})'

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
        self.replaced_log_importance_weights = {} # Used for computing the weights
        self.rs_entries = []
        self.rs_entries_dict_address_base = {}
        self.rs_entries_dict_address = {}
        self._rs_stack = RejectionSamplingStack()

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
            self.log_prob_observed += torch.sum(variable.log_prob)
            if variable.name:
                self.variables_observed[variable.name] = variable
            else:
                self.variables_observed[variable.address] = variable
        self.variables_dict_address[variable.address] = variable
        self.variables_dict_address_base[variable.address_base] = variable
        if variable.observed or variable.control:
            self.log_prob += torch.sum(variable.log_prob)
        if variable.log_importance_weight is not None:
            self.log_importance_weight += variable.log_importance_weight

    def add_rs_entry(self, entry):
        self.rs_entries.append(entry) # TODO: is this really needed anymore (even though rs_entries_dict_address exists)?
        self.rs_entries_dict_address_base[entry.address_base] = entry
        self.rs_entries_dict_address[entry.address] = entry

    def end(self, result, execution_time_sec):
        self.result = result
        self.execution_time_sec = execution_time_sec
        self.length = len(self.variables)

    def refresh_weights_and_dictionaries(self):
        # Compute weights and re-construct dictionaries
        self.variables_dict_address = {}
        self.variables_dict_address_base = {}
        self.log_prob_observed = 0.
        self.log_prob = 0.
        self.log_importance_weight = 0.
        for variable in self.variables:
            if variable.observed:
                if variable.name:
                    self.variables_observed[variable.name] = variable
                else:
                    self.variables_observed[variable.address] = variable
            self.variables_dict_address[variable.address] = variable
            self.variables_dict_address_base[variable.address_base] = variable

            # Re-compute weights
            if variable.observed or variable.control:
                self.log_prob_observed += torch.sum(variable.log_prob)
                self.log_prob += torch.sum(variable.log_prob)
            if variable.log_importance_weight is not None:
                self.log_importance_weight += variable.log_importance_weight

        for rs_entry in self.rs_entries_dict_address.values():
            if rs_entry.log_importance_weight is not None:
                self.log_importance_weight += rs_entry.log_importance_weight
                #self.log_prob += rs_entry.log_prob

        self.length = len(self.variables)

    def discard_rejected(self):
        self.variables = list(filter(lambda v: v.accepted, self.variables))
        self.refresh_weights_and_dictionaries()

    def last_instance(self, address_base):
        if address_base in self.variables_dict_address_base:
            return self.variables_dict_address_base[address_base].instance
        else:
            return 0

    def last_rs_entry_instance(self, address_base):
        if address_base in self.rs_entries_dict_address_base:
            return self.rs_entries_dict_address_base[address_base].instance
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
