import torch
import sys
import opcode
import random
import time
import warnings

from .distributions import Normal, Categorical, Uniform, TruncatedNormal, Factor
from .trace import Variable, Trace
from . import util, TraceMode, PriorInflation, InferenceEngine


_trace_mode = TraceMode.PRIOR
_inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
_prior_inflation = PriorInflation.DISABLED
_likelihood_importance = 1.
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_inference_network_proposal_min_train_iterations = None
_current_trace_previous_variable = None
_current_trace_observed_variables = None
_current_trace_execution_start = None
_metropolis_hastings_trace = None
_metropolis_hastings_site_address = None
_metropolis_hastings_site_transition_log_prob = 0
_address_dictionary = None


# _extract_address and _extract_target_of_assignment code by Tobias Kohn (kohnt@tobiaskohn.ch)
def _extract_address(root_function_name):
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
    return '{}__{}'.format(ip, '__'.join(reversed(names)))


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
    if _current_trace is None:
        return
    if address is None:
        address_base = _extract_address(_current_trace_root_function_name) + '__None'
    else:
        address_base = address + '__None'
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    variable = Variable(distribution=None, value=value, address_base=address_base, address=address, instance=instance, log_prob=0., tagged=True, name=name)
    _current_trace.add(variable)


def factor(log_prob=None, log_prob_func=None, name=None, address=None):
    dist = Factor(log_prob=log_prob, log_prob_func=log_prob_func)
    observe(dist, name=name, address=address)


def observe(distribution, value=None, name=None, address=None):
    global _current_trace
    if _current_trace is None:
        return
    if address is None:
        address_base = _extract_address(_current_trace_root_function_name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    if name in _current_trace_observed_variables:
        # Override observed value
        value = _current_trace_observed_variables[name]
    elif value is not None:
        value = util.to_tensor(value)
    elif _trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK and distribution is not None:
        value = distribution.sample()
    else:
        value = None

    if value is None and not isinstance(distribution, Factor):
        observed = False
        log_prob = None
        log_importance_weight = None
    else:
        observed = True
        log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            log_importance_weight = float(log_prob)
        else:
            log_importance_weight = None  # TODO: Check the reason/behavior of this

    variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, observed=observed, name=name)
    _current_trace.add(variable)
    return variable.value


def sample(distribution, name=None, address=None, control=True):
    global _current_trace
    global _current_trace_previous_variable

    if _current_trace is None:
        return distribution.sample()

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        control = True

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)

    instance = _current_trace.last_instance(address_base) + 1

    if name in _current_trace_observed_variables:
        # Variable is observed
        address = address_base + '__' + str(instance)
        value = _current_trace_observed_variables[name]
        log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            log_importance_weight = float(log_prob)
        else:
            log_importance_weight = None  # TODO: Check the reason/behavior for this
        variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, observed=True, name=name)
    else:
        # Variable is sampled
        reused = False
        observed = False
        if _trace_mode == TraceMode.POSTERIOR:
            if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                address = address_base + '__' + str(instance)
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
                address = address_base + '__' + str(instance)
                if control:
                    variable = Variable(distribution=distribution, value=None, address_base=address_base, address=address, instance=instance, log_prob=0., control=control, name=name, observed=observed, reused=reused)
                    proposal_distribution = _current_trace_inference_network._infer_step(variable, prev_variable=_current_trace_previous_variable, proposal_min_train_iterations=_current_trace_inference_network_proposal_min_train_iterations)
                    value = proposal_distribution.sample()
                    if value.dim() > 0:
                        value = value[0]
                    log_prob = distribution.log_prob(value, sum=True)
                    proposal_log_prob = proposal_distribution.log_prob(value, sum=True)
                    if util.has_nan_or_inf(log_prob):
                        warnings.warn('Prior log_prob has NaN, inf, or -inf.\ndistribution: {}\n value: {}\n log_prob: {}'.format(distribution, value, log_prob))
                    if util.has_nan_or_inf(proposal_log_prob):
                        warnings.warn('Proposal log_prob has NaN, inf, or -inf.\ndistribution: {}\n value: {}\nlog_prob: {}'.format(proposal_distribution, value, proposal_log_prob))
                    log_importance_weight = float(log_prob) - float(proposal_log_prob)
                    variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, control=control, name=name, observed=observed, reused=reused)
                    _current_trace_previous_variable = variable
                    # print('prev_var address {}'.format(variable.address))
                else:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                    log_importance_weight = None
            else:  # _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
                address = address_base + '__' + str(instance)
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
            address = address_base + '__' + str(instance)
            inflated_distribution = _inflate(distribution)
            if inflated_distribution is None:
                value = distribution.sample()
                log_prob = distribution.log_prob(value, sum=True)
                log_importance_weight = None
            else:
                value = inflated_distribution.sample()
                log_prob = distribution.log_prob(value, sum=True)
                log_importance_weight = float(log_prob) - float(inflated_distribution.log_prob(value, sum=True))  # To account for prior inflation

        variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, control=control, name=name, observed=observed, reused=reused)

    _current_trace.add(variable)
    return variable.value


def _init_traces(func, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, address_dictionary=None, likelihood_importance=1.):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _likelihood_importance
    _trace_mode = trace_mode
    _inference_engine = inference_engine
    _prior_inflation = prior_inflation
    _likelihood_importance = likelihood_importance
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
        if any([v is None for v in observe.values()]):
            raise RuntimeError('Observe has missing value(s): {}'.format(observe))
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
            variable = random.choice(_metropolis_hastings_trace.variables_controlled)
            _metropolis_hastings_site_address = variable.address


def _begin_trace():
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_execution_start
    _current_trace_execution_start = time.time()
    _current_trace = Trace()
    _current_trace_previous_variable = None


def _end_trace(result):
    global _current_trace
    execution_time_sec = time.time() - _current_trace_execution_start
    _current_trace.end(result, execution_time_sec)
    trace = _current_trace
    _current_trace = None
    return trace
