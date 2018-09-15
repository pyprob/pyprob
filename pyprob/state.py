import torch
import sys
import opcode
import random
import time
from termcolor import colored

from .distributions import Normal, Categorical, Uniform, TruncatedNormal
from .trace import Variable, Trace
from . import util, TraceMode, PriorInflation, InferenceEngine


_trace_mode = TraceMode.PRIOR
_inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
_prior_inflation = PriorInflation.DISABLED
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_previous_variable = None
_current_trace_replaced_variable_proposal_distributions = {}
_current_trace_observed_variables = None
_current_trace_execution_start = None
_metropolis_hastings_trace = None
_metropolis_hastings_site_address = None
_metropolis_hastings_site_transition_log_prob = 0


# extract_address and _extract_target_of_assignment code by Tobias Kohn (kohnt@tobiaskohn.ch)
def extract_address(root_function_name):
    # Retun an address in the format:
    # 'instruction pointer' / 'qualified function name'
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
    return "{}/{}".format(ip, '/'.join(reversed(names)))


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


def _sample_with_prior_inflation(distribution):
    if _prior_inflation == PriorInflation.ENABLED:
        if isinstance(distribution, Categorical):
            distribution = Categorical(util.to_tensor(torch.zeros(distribution.num_categories).fill_(1./distribution.num_categories)))
        elif isinstance(distribution, Normal):
            distribution = Normal(distribution.mean, distribution.stddev * 3)
    return distribution.sample()


def observe(distribution=None, value=None, name=None, address=None):
    global _current_trace
    if address is None:
        address_base = extract_address(_current_trace_root_function_name)
    else:
        address_base = address
    instance = _current_trace.last_instance(address_base) + 1
    address_suffix = 'None' if distribution is None else distribution._address_suffix
    address = '{}_{}_{}'.format(address_base, address_suffix, instance)

    if name in _current_trace_observed_variables:
        # Override observed value
        value = _current_trace_observed_variables[name]
    elif value is not None:
        value = util.to_tensor(value)
    elif distribution is not None:
        value = distribution.sample()

    if distribution is None or value is None:
        log_prob = 0.
    else:
        log_prob = distribution.log_prob(value, sum=True)
    if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
        _current_trace.log_importance_weight += log_prob

    variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, observed=True, name=name)
    _current_trace.add(variable)


def sample(distribution, control=True, replace=False, name=None, address=None):
    global _current_trace

    # Only replace if controlled
    if not control:
        replace = False

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        control = True
        replace = False

    if address is None:
        address_base = extract_address(_current_trace_root_function_name)
    else:
        address_base = address
    instance = _current_trace.last_instance(address_base) + 1
    address = '{}_{}_{}'.format(address_base, distribution._address_suffix, 'replaced' if replace else str(instance))

    if name in _current_trace_observed_variables:
        # Variable is observed
        value = _current_trace_observed_variables[name]
        log_prob = distribution.log_prob(value, sum=True)
        _current_trace.log_importance_weight += log_prob.item()
        variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, observed=True, name=name)
    else:
        reused = False
        observed = False
        update_previous_variable = False

        if _trace_mode == TraceMode.PRIOR:
            value = _sample_with_prior_inflation(distribution)
            log_prob = distribution.log_prob(value, sum=True)
        else:  # _trace_mode == TraceMode.POSTERIOR
            if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                value = distribution.sample()
                log_prob = distribution.log_prob(value, sum=True)
                # _current_trace.log_importance_weight += 0  # Not computed because log_importance_weight is zero when running importance sampling with prior as proposal
            elif _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                if control:
                    global _current_trace_previous_variable
                    global _current_trace_replaced_variable_proposal_distributions
                    _current_trace_inference_network.eval()
                    variable = Variable(distribution=distribution, value=None, address_base=address_base, address=address, instance=instance, log_prob=0., control=control, replace=replace, name=name, observed=observed, reused=reused)
                    if replace:
                        if address not in _current_trace_replaced_variable_proposal_distributions:
                            _current_trace_replaced_variable_proposal_distributions[address] = _current_trace_inference_network.infer_trace_step(variable, previous_variable=_current_trace_previous_variable)
                            update_previous_variable = True
                        proposal_distribution = _current_trace_replaced_variable_proposal_distributions[address]
                    else:
                        proposal_distribution = _current_trace_inference_network.infer_trace_step(variable, previous_variable=_current_trace_previous_variable)
                        update_previous_variable = True

                    value = proposal_distribution.sample()[0]
                    log_prob = distribution.log_prob(value, sum=True)
                    proposal_log_prob = proposal_distribution.log_prob(value, sum=True)
                    if util.has_nan_or_inf(log_prob):
                        print(colored('Warning: prior log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                        print('distribution', distribution)
                        print('value', value)
                        print('log_prob', log_prob)
                    if util.has_nan_or_inf(proposal_log_prob):
                        print(colored('Warning: proposal log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                        print('distribution', proposal_distribution)
                        print('value', value)
                        print('log_prob', proposal_log_prob)
                    _current_trace.log_importance_weight += log_prob - proposal_log_prob
                else:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
            else:  # _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
                if _metropolis_hastings_trace is None:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                else:
                    if address == _metropolis_hastings_site_address:
                        global _metropolis_hastings_site_transition_log_prob
                        _metropolis_hastings_site_transition_log_prob = util.to_tensor(0.)
                        if _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
                            if isinstance(distribution, Normal):
                                proposal_kernel_func = lambda x: Normal(x, 1)
                            elif isinstance(distribution, Uniform):
                                proposal_kernel_func = lambda x: TruncatedNormal(x, 0.1, low=distribution.low, high=distribution.high)
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

        variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, control=control, replace=replace, name=name, observed=observed, reused=reused)
        if update_previous_variable:
            _current_trace_previous_variable = variable

    _current_trace.add(variable)
    return variable.value


def begin_trace(func, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    _trace_mode = trace_mode
    _inference_engine = inference_engine
    _prior_inflation = prior_inflation
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    global _current_trace_replaced_variable_proposal_distributions
    global _current_trace_observed_variables
    global _current_trace_execution_start
    _current_trace_execution_start = time.time()
    _current_trace = Trace()
    _current_trace_root_function_name = func.__code__.co_name
    _current_trace_replaced_variable_proposal_distributions = {}
    if observe is None:
        _current_trace_observed_variables = {}
    else:
        _current_trace_observed_variables = observe
    _current_trace_inference_network = inference_network
    if _current_trace_inference_network is None:
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            raise ValueError('Cannot run trace with IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK without an inference network.')
    else:
        _current_trace_inference_network.infer_trace_init(_current_trace_observed_variables)

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        global _metropolis_hastings_trace
        global _metropolis_hastings_site_transition_log_prob
        _metropolis_hastings_trace = metropolis_hastings_trace
        _metropolis_hastings_site_transition_log_prob = None
        if _metropolis_hastings_trace is not None:
            global _metropolis_hastings_site_address
            variable = random.choice(_metropolis_hastings_trace.variables_controlled)
            _metropolis_hastings_site_address = variable.address


def end_trace(result):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    _inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
    _prior_inflation = PriorInflation.DISABLED
    execution_time_sec = time.time() - _current_trace_execution_start
    _current_trace.end(result, execution_time_sec)
    ret = _current_trace
    _current_trace = None
    _current_trace_root_function_name = None
    _current_trace_inference_network = None
    return ret
