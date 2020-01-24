import torch
import sys
import opcode
import random
import time
from termcolor import colored

from .distributions import Normal, Categorical, Uniform, TruncatedNormal
from .trace import Variable, RSEntry, Trace
from . import util, TraceMode, PriorInflation, InferenceEngine,ImportanceWeighting
from .nn import InferenceNetworkLSTM


_trace_mode = TraceMode.PRIOR
_inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
_prior_inflation = PriorInflation.DISABLED
_likelihood_importance = 1.
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_inference_network_proposal_min_train_iterations = None
_current_trace_previous_variable = None
_current_trace_replaced_variable_proposal_distributions = {}
_current_trace_observed_variables = None
_current_trace_execution_start = None
_metropolis_hastings_trace = None
_metropolis_hastings_site_address = None
_metropolis_hastings_site_transition_log_prob = 0
_address_dictionary = None
_variables_observed_inf_training = []
_importance_weighting = ImportanceWeighting.IW2


class RejectionEndException(Exception):
    def __init__(self, rs_entry):
        self.length = int(rs_entry.iteration.item()) + 1

class RSPartialTrace:
    def __init__(self, trace=None, rejection_address=None):
        self.trace = trace
        self.rejection_address = rejection_address
        self.index = 0
        if trace is None:
            if rejection_address is not None:
                raise ValueError('In a partial trace, when trace=None, rejection_address should also be None')
            self._done = True
        else:
            if rejection_address is None:
                raise ValueError('rejection_address cannot be None in a partial trace')
            self._done = False

    def get_variable(self, address):
        # Returns the next variable from the partial trace
        # Verifies the variable address to be the same as the given address
        # Updates the partial trace (advances the index and sets it to null if reached the target address)
        variable = self.trace.variables[self.index]
        assert variable.address == address
        self.index += 1
        return variable

    def rs_checkpoint(self, address):
        # It is called once a rs_start is reached. Input is the address of this rs_start.
        # Will mark the partial trace as done if the address matches the rejection address in the partial trace
        if address == self.rejection_address:
            self._done = True

    @property
    def done(self):
        return self._done

_current_rs_partial_trace = None


# _extract_address and _extract_target_of_assignment code by Tobias Kohn (kohnt@tobiaskohn.ch)
def _extract_address(root_function_name, user_specified_name):
    # Retun an address in the format:
    # 'instruction pointer' __ 'qualified function name'
    frame = sys._getframe(2)
    ip = frame.f_lasti
    names = []
    var_name = _extract_target_of_assignment()
    if var_name is None:
        names.append('?')
    else:
        names.append(var_name)
    while frame is not None:
        n = frame.f_code.co_name
        if n.startswith('<') and not n == '<listcomp>':
            break
        names.append(n)
        if n == root_function_name:
            break
        frame = frame.f_back
    address_base_noname = '{}__{}'.format(ip, '__'.join(reversed(names)))
    return '{}__{}'.format(address_base_noname, user_specified_name)


def _extract_target_of_assignment():
    frame = sys._getframe(3)
    code = frame.f_code
    next_instruction = code.co_code[frame.f_lasti+2]
    instruction_arg = code.co_code[frame.f_lasti+3]
    instruction_name = opcode.opname[next_instruction]
    if instruction_name == 'STORE_FAST':
        return code.co_varnames[instruction_arg]
    elif instruction_name in ['STORE_NAME', 'STORE_GLOBAL']:
        return code.co_names[instruction_arg]
    elif instruction_name in ['LOAD_FAST', 'LOAD_NAME', 'LOAD_GLOBAL'] and \
            opcode.opname[code.co_code[frame.f_lasti+4]] in ['LOAD_CONST', 'LOAD_FAST'] and \
            opcode.opname[code.co_code[frame.f_lasti+6]] == 'STORE_SUBSCR':
        base_name = (code.co_varnames if instruction_name == 'LOAD_FAST' else code.co_names)[instruction_arg]
        second_instruction = opcode.opname[code.co_code[frame.f_lasti+4]]
        second_arg = code.co_code[frame.f_lasti+5]
        if second_instruction == 'LOAD_CONST':
            value = code.co_consts[second_arg]
        elif second_instruction == 'LOAD_FAST':
            var_name = code.co_varnames[second_arg]
            value = frame.f_locals[var_name]
        else:
            value = None
        if type(value) is int:
            index_name = str(value)
            return base_name + '[' + index_name + ']'
        else:
            return None
    elif instruction_name == 'RETURN_VALUE':
        return 'return'
    else:
        return None


def _inflate(distribution):
    if _prior_inflation == PriorInflation.ENABLED:
        if isinstance(distribution, Categorical):
            return Categorical(util.to_tensor(torch.zeros(distribution.num_categories).fill_(1./distribution.num_categories)))
        elif isinstance(distribution, Normal):
            return Normal(distribution.mean, distribution.stddev * 3)
    return None


def tag(value, name=None, address=None):
    global _current_trace
    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__None'
    else:
        address_base = address + '__None'
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    value = util.to_tensor(value)

    variable = Variable(distribution=None, value=value,
                        address_base=address_base, address=address, instance=instance, log_prob=0.,
                        tagged=True, name=name)
    _current_trace.add(variable)


def observe(distribution, value=None, constants={}, name=None, address=None):
    global _current_trace

    if not _current_trace._rs_stack.isempty():
        raise ValueError('No observation can be within a rejection sampling loop')

    # make values in constants tensors
    constants = util.constants_to_tensors(constants)

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    if not _current_rs_partial_trace.done:
        variable = _current_rs_partial_trace.get_variable(address)
    else:
        if name in _current_trace_observed_variables:
            # Override observed value
            value = _current_trace_observed_variables[name]
        elif value is not None:
            value = util.to_tensor(value) # TODO: doesn't it break training?
        elif distribution is not None:
            value = distribution.sample()

        log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            log_importance_weight = float(log_prob)
        else:
            log_importance_weight = None  # TODO: Check the reason/behavior for this

        variable = Variable(distribution=distribution, value=value, constants=constants,
                            address_base=address_base, address=address, instance=instance,
                            log_prob=log_prob, log_importance_weight=log_importance_weight,
                            observed=True, name=name)
    _current_trace.add(variable)
    return variable.value


def sample(distribution, constants={}, control=True, replace=False, name=None,
           address=None):
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_replaced_variable_proposal_distributions

    # make values in constants tensors
    constants = util.constants_to_tensors(constants)

    replace = False
    if (not _current_trace._rs_stack.isempty()):
        # If there is an active rejection sampling,
        # -> the variable is "replaced"
        # -> rejection_address is set to the address of the latest started rejection sampling
        replace = True
        if _current_trace._rs_stack.top_entry.control == False:
            # An uncontrolled rejection sampling makes everything under its scope uncontrolled.
            control = False

    # Only replace if controlled
    if not control:
        replace = False

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        control = True
        replace = False

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)

    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    if not _current_rs_partial_trace.done:
        variable = _current_rs_partial_trace.get_variable(address)
    else:
        if name in _variables_observed_inf_training + list(_current_trace_observed_variables):
            if not _current_trace._rs_stack.isempty():
                raise ValueError('No observation can be within a rejection sampling loop')

        if name in _current_trace_observed_variables:
            # Variable is observed
            value = _current_trace_observed_variables[name]
            log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
            if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                log_importance_weight = float(log_prob)
            else:
                log_importance_weight = None  # TODO: Check the reason/behavior for this
            variable = Variable(distribution=distribution, value=value, constants=constants,
                                address_base=address_base, address=address, instance=instance,
                                log_prob=log_prob, log_importance_weight=log_importance_weight,
                                observed=True, name=name)
        else:
            # Variable is sampled
            reused = False
            if name in _variables_observed_inf_training:
                observed = True
            else:
                observed = False

            if _trace_mode == TraceMode.POSTERIOR:
                if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                    inflated_distribution = _inflate(distribution)
                    if inflated_distribution is None:
                        value = distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                        log_importance_weight = None
                    else:
                        value = inflated_distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                        log_importance_weight = float(log_prob) - float(inflated_distribution.log_prob(value, sum=True))  # To account for prior inflation
                elif _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                    address = address_base + '__' + ('replaced' if replace else str(instance))  # Address seen by inference network
                    variable = Variable(distribution=distribution, value=None,
                                        constants=constants,
                                        address_base=address_base,
                                        address=address, instance=instance,
                                        log_prob=0., control=control,
                                        replace=replace, name=name,
                                        observed=observed, reused=reused)
                    proposal_distribution = _current_trace_inference_network._infer_step(variable,
                                                                                        prev_variable=_current_trace_previous_variable,
                                                                                        proposal_min_train_iterations=_current_trace_inference_network_proposal_min_train_iterations)
                    if replace and _importance_weighting == ImportanceWeighting.IW0: # use prior as proposal for all addresses with replace=True
                        proposal_distribution = distribution
                    value = proposal_distribution.sample()
                    if distribution.name == "Normal":
                        value = value.view(torch.Size([-1]) + distribution.loc_shape)

                    # removes the redundant batch dimension!
                    # maybe rethink this step
                    if value.dim() > 0:
                        value = value.squeeze(0)

                    log_prob = distribution.log_prob(value, sum=True)
                    proposal_log_prob = proposal_distribution.log_prob(value, sum=True)
                    if util.has_nan_or_inf(log_prob):
                        print(colored('Warning: prior log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                        print(address)
                        print('distribution', distribution)
                        print('value', value)
                        print('log_prob', log_prob)
                    if util.has_nan_or_inf(proposal_log_prob):
                        print(colored('Warning: proposal log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                        print('distribution', proposal_distribution)
                        print('value', value)
                        print('log_prob', proposal_log_prob)
                    log_importance_weight = float(log_prob) - float(proposal_log_prob)
                    variable = Variable(distribution=distribution,
                                        value=value, constants=constants,
                                        address_base=address_base,
                                        address=address, instance=instance,
                                        log_prob=log_prob,
                                        log_importance_weight=log_importance_weight,
                                        control=control, replace=replace,
                                        name=name, observed=observed,
                                        reused=reused)
                    _current_trace_previous_variable = variable
                    # print('prev_var address {}'.format(variable.address))
                    address = address_base + '__' + str(instance)
                else:  # _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
                    log_importance_weight = None
                    if _metropolis_hastings_trace is None:
                        value = distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                    else:
                        if address == _metropolis_hastings_site_address:
                            global _metropolis_hastings_site_transition_log_prob
                            _metropolis_hastings_site_transition_log_prob = util.to_tensor(0.)
                            if _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
                                if isinstance(distribution, Normal):
                                    proposal_kernel_func = lambda x: Normal(x, distribution.stddev)
                                elif isinstance(distribution, Uniform):
                                    proposal_kernel_func = lambda x: TruncatedNormal(x, 0.1*(distribution.high - distribution.low), low=distribution.low, high=distribution.high)
                                else:
                                    proposal_kernel_func = None

                                if proposal_kernel_func is not None:
                                    _metropolis_hastings_site_value = _metropolis_hastings_trace.variables_dict_address[address].value
                                    _metropolis_hastings_site_log_prob = _metropolis_hastings_trace.variables_dict_address[address].log_prob
                                    proposal_kernel_forward = proposal_kernel_func(_metropolis_hastings_site_value)
                                    alpha = 0.5
                                    if random.random() < alpha:
                                        value = proposal_kernel_forward.sample()
                                    else:
                                        value = distribution.sample()
                                    log_prob = distribution.log_prob(value, sum=True)
                                    proposal_kernel_reverse = proposal_kernel_func(value)

                                    _metropolis_hastings_site_transition_log_prob = torch.log(alpha * torch.exp(proposal_kernel_reverse.log_prob(_metropolis_hastings_site_value, sum=True)) + (1 - alpha) * torch.exp(_metropolis_hastings_site_log_prob)) + log_prob
                                    _metropolis_hastings_site_transition_log_prob -= torch.log(alpha * torch.exp(proposal_kernel_forward.log_prob(value, sum=True)) + (1 - alpha) * torch.exp(log_prob)) + _metropolis_hastings_site_log_prob
                                else:
                                    value = distribution.sample()
                                    log_prob = distribution.log_prob(value, sum=True)
                            else:
                                value = distribution.sample()
                                log_prob = distribution.log_prob(value, sum=True)
                            reused = False
                        elif address not in _metropolis_hastings_trace.variables_dict_address:
                            value = distribution.sample()
                            log_prob = distribution.log_prob(value, sum=True)
                            reused = False
                        else:
                            value = _metropolis_hastings_trace.variables_dict_address[address].value
                            reused = True
                            try:  # Takes care of issues such as changed distribution parameters (e.g., batch size) that prevent a rescoring of a reused value under this distribution.
                                log_prob = distribution.log_prob(value, sum=True)
                            except:
                                value = distribution.sample()
                                log_prob = distribution.log_prob(value, sum=True)
                                reused = False

            else:  # _trace_mode == TraceMode.PRIOR or _trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
                if _trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
                    address = address_base + '__' + ('replaced' if replace else str(instance))
                inflated_distribution = _inflate(distribution)
                if inflated_distribution is None:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                    log_importance_weight = None
                else:
                    value = inflated_distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                    log_importance_weight = float(log_prob) - float(inflated_distribution.log_prob(value, sum=True))  # To account for prior inflation

            if _trace_mode == TraceMode.POSTERIOR and _importance_weighting == ImportanceWeighting.IW2:
                # IW2 should take all the weights into account => no replaced variables
                replace=False

            variable = Variable(distribution=distribution, value=value,
                                constants=constants, address_base=address_base,
                                address=address, instance=instance,
                                log_prob=log_prob,
                                log_importance_weight=log_importance_weight,
                                control=control, replace=replace, name=name,
                                observed=observed, reused=reused)

    _current_trace.add(variable)
    return variable.value


def _set_observed_from_inf(variables_observed_inf_training):
    global _variables_observed_inf_training
    _variables_observed_inf_training = variables_observed_inf_training


def rs_start(control=True, name=None, address=None):
    global _current_trace_previous_variable

    rejection_sampling_suffix = 'rejsmp'

    # Compute the address and address_base
    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__' + rejection_sampling_suffix
        # Problematic for nested rejection sampling!
    else:
        address_base = address + '__' + rejection_sampling_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)

    # Compute iteration number
    if (not _current_trace._rs_stack.isempty()) and _current_trace._rs_stack.top_entry.address_base == address_base:
        # It is not a new rejection sampling. Rather, it's retrying to sample from the base distribution
        # We use the same instance number in such cases
        last_entry = _current_trace._rs_stack.top_entry
        instance = last_entry.instance
        iteration = last_entry.iteration + 1
    else:
        instance = _current_trace.last_rs_entry_instance(address_base) + 1
        iteration = util.to_tensor(0)

    address = address_base + '__' + str(instance)

    if not _current_rs_partial_trace.done:
        _current_rs_partial_trace.rs_checkpoint(address)

    entry = RSEntry(address=address, address_base=address_base, name=name, instance=instance, control=control, iteration=iteration)

    _current_trace.add_rs_entry(entry)
    previous_variable = None if len(_current_trace.variables) == 0 else _current_trace.variables[-1]
    if iteration == 0:
        # Start of a new rejection sampling -> should be pushed to the stack
        hidden_state = None
        if _current_trace_inference_network is not None and isinstance(_current_trace_inference_network, InferenceNetworkLSTM):
            # TODO: also check that inference engine is IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK and importance_weighting is 
            hidden_state = _current_trace_inference_network._infer_lstm_state #TODO: clone/copy
        _current_trace._rs_stack.push(entry=entry,
                                      previous_variable=previous_variable,
                                      network_state=(hidden_state, _current_trace_previous_variable))
    else:
        ## Retrying the same rejection sampling loop
        # Set accepted=False for all the variables samples since the last rs_start with the same address
        target_variable = _current_trace._rs_stack.top_previous_variable
        for i in range(len(_current_trace.variables)-1, -1, -1):
            variable = _current_trace.variables[i]
            if variable is target_variable:
                break
            variable.accepted = False

        # Update trace's previous variable and restore LSTM's hidden state (if exists)
        (hidden_state, network_previous_state) = _current_trace._rs_stack.top_network_state
        _current_trace_previous_variable = network_previous_state
        if hidden_state is not None:
            _current_trace_inference_network._infer_lstm_state = hidden_state
        # Replace the active rejection sampling entry and previous variable (hidden remains the same)
        _current_trace._rs_stack.updateTop(entry, previous_variable)

def rs_end():
    if _current_rs_partial_trace.rejection_address == _current_trace._rs_stack.top_entry.address:
        raise RejectionEndException(_current_trace._rs_stack.top_entry)
    _current_trace._rs_stack.pop()
    # TODO: add a dummy variable or a tag to the trace?


def _init_traces(func, trace_mode=TraceMode.PRIOR,
                 prior_inflation=PriorInflation.DISABLED,
                 inference_engine=InferenceEngine.IMPORTANCE_SAMPLING,
                 inference_network=None, observe=None,
                 metropolis_hastings_trace=None, address_dictionary=None,
                 likelihood_importance=1., importance_weighting=ImportanceWeighting.IW2):

    """ Initialize the trace object

    Inputs:

    observe -- (Key, Value) = (name, observed value)
    """

    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _likelihood_importance
    global _importance_weighting
    _trace_mode = trace_mode
    _inference_engine = inference_engine
    _prior_inflation = prior_inflation
    _likelihood_importance = likelihood_importance
    _importance_weighting = importance_weighting
    global _current_trace_root_function_name
    global _current_trace_inference_network
    global _current_trace_inference_network_proposal_min_train_iterations
    global _current_trace_observed_variables
    global _address_dictionary

    _address_dictionary = address_dictionary
    _current_trace_root_function_name = func.__code__.co_name
    if observe is None:
        _current_trace_observed_variables = {}
    else:
        _current_trace_observed_variables = observe

    _current_trace_inference_network = inference_network
    if _current_trace_inference_network is None:
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            raise ValueError('Cannot run trace with IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK without an inference network.')
    else:
        _current_trace_inference_network.eval()
        _current_trace_inference_network._infer_init(_current_trace_observed_variables)
        # _current_trace_inference_network_proposal_min_train_iterations = int(_current_trace_inference_network._total_train_iterations / 10)
        _current_trace_inference_network_proposal_min_train_iterations = None

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        global _metropolis_hastings_trace
        global _metropolis_hastings_site_transition_log_prob
        _metropolis_hastings_trace = metropolis_hastings_trace
        _metropolis_hastings_site_transition_log_prob = None
        if _metropolis_hastings_trace is not None:
            global _metropolis_hastings_site_address
            variable = random.choice([v for v in _metropolis_hastings_trace.variables if v.control])
            _metropolis_hastings_site_address = variable.address


def _begin_trace(rs_partial_trace=RSPartialTrace()):
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_replaced_variable_proposal_distributions
    global _current_trace_execution_start
    global _current_rs_partial_trace

    _current_trace_execution_start = time.time()
    _current_trace = Trace()
    _current_trace_previous_variable = None
    _current_trace_replaced_variable_proposal_distributions = {}

    _current_rs_partial_trace = rs_partial_trace


def _end_trace(result):
    # Make sure there is no active rejection sampling remaining.
    if not _current_trace._rs_stack.isempty():
        print(f'{_current_trace._rs_stack.size()}, {_current_trace._rs_stack.top_entry.address}')
        raise Exception(f'Trace ended while active rejection samplings still exist')

    execution_time_sec = time.time() - _current_trace_execution_start
    _current_trace.end(result, execution_time_sec)

    return _current_trace
