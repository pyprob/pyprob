import sys
import opcode
import random
from termcolor import colored
import torch

from .distributions import Uniform, Normal, TruncatedNormal, Categorical
from .trace import Sample, Trace
from . import util, TraceMode, PriorInflation, InferenceEngine

_trace_mode = TraceMode.NONE
_inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
_prior_inflation = PriorInflation.DISABLED
_observation_importance_exponent = 1.
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_previous_sample = None
_current_trace_replaced_sample_proposal_distributions = {}
_metropolis_hastings_trace = None
_metropolis_hastings_site_address = None
_metropolis_hastings_site_transition_log_prob = 0


def extract_address(root_function_name):
    # tb = traceback.extract_stack()
    # print()
    # for t in tb:
    #     print(t[0], t[1], t[2], t[3])
    # frame = tb[-3]
    # return '{0}/{1}/{2}'.format(frame[1], frame[2], frame[3])
    # return '{0}/{1}'.format(frame[1], frame[2])
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
            distribution = Categorical(util.to_variable(torch.zeros(distribution.length_categories).fill_(1./distribution.length_categories)))
        elif isinstance(distribution, Normal):
            distribution = Normal(distribution.mean, distribution.stddev * 3)
    return distribution.sample()


def sample(distribution, control=True, replace=False, address=None):
    if _trace_mode == TraceMode.NONE:
        return _sample_with_prior_inflation(distribution)  # Forward sample
    else:  # _trace_mode == TraceMode.PRIOR or _trace_mode == TraceMode.POSTERIOR
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
        address = '{}_{}_{}'.format(address_base, distribution.address_suffix, 'replaced' if replace else str(instance))
        reused = False
        update_previous_sample = False

        if _trace_mode == TraceMode.PRIOR:
            value = _sample_with_prior_inflation(distribution)
            log_prob = distribution.log_prob(value)
        else:  # _trace_mode == TraceMode.POSTERIOR
            if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                value = distribution.sample()
                log_prob = distribution.log_prob(value)
                # _current_trace.log_importance_weight += 0  # Not computed because log_importance_weight is zero when running importance sampling with prior as proposal
            elif _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                if control:
                    global _current_trace_previous_sample
                    global _current_trace_replaced_sample_proposal_distributions
                    _current_trace_inference_network.eval()
                    current_sample = Sample(distribution, 0, address_base, address, instance, log_prob=0, control=control, replace=replace)
                    if replace:
                        if address not in _current_trace_replaced_sample_proposal_distributions:
                            _current_trace_replaced_sample_proposal_distributions[address] = _current_trace_inference_network.forward_one_time_step(_current_trace_previous_sample, current_sample)
                            update_previous_sample = True
                        proposal_distribution = _current_trace_replaced_sample_proposal_distributions[address]
                    else:
                        proposal_distribution = _current_trace_inference_network.forward_one_time_step(_current_trace_previous_sample, current_sample)
                        update_previous_sample = True

                    value = proposal_distribution.sample()[0]
                    log_prob = distribution.log_prob(value)
                    proposal_log_prob = proposal_distribution.log_prob(value)
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
                    _current_trace.log_importance_weight += log_prob - proposal_log_prob.item()
                else:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value)
            else:  # _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
                if _metropolis_hastings_trace is None:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value)
                else:
                    if address == _metropolis_hastings_site_address:
                        global _metropolis_hastings_site_transition_log_prob
                        _metropolis_hastings_site_transition_log_prob = util.to_variable(0.)
                        if _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
                            if isinstance(distribution, Normal):
                                proposal_kernel_func = lambda x:Normal(x, 1)
                            elif isinstance(distribution, Uniform):
                                proposal_kernel_func = lambda x:TruncatedNormal(x, 0.1, low=distribution.low, high=distribution.high)
                            else:
                                proposal_kernel_func = None

                            if proposal_kernel_func is not None:
                                _metropolis_hastings_site_value = _metropolis_hastings_trace._samples_all_dict_address[address].value
                                _metropolis_hastings_site_log_prob = _metropolis_hastings_trace._samples_all_dict_address[address].log_prob
                                proposal_kernel_forward = proposal_kernel_func(_metropolis_hastings_site_value)
                                alpha = 0.5
                                if random.random() < alpha:
                                    value = proposal_kernel_forward.sample()
                                else:
                                    value = distribution.sample()
                                log_prob = distribution.log_prob(value)
                                proposal_kernel_reverse = proposal_kernel_func(value)

                                _metropolis_hastings_site_transition_log_prob = torch.log(alpha * torch.exp(proposal_kernel_reverse.log_prob(_metropolis_hastings_site_value)) + (1 - alpha) * torch.exp(_metropolis_hastings_site_log_prob)) + log_prob
                                _metropolis_hastings_site_transition_log_prob -= torch.log(alpha * torch.exp(proposal_kernel_forward.log_prob(value)) + (1 - alpha) * torch.exp(log_prob)) + _metropolis_hastings_site_log_prob
                            else:
                                value = distribution.sample()
                                log_prob = distribution.log_prob(value)
                        else:
                            value = distribution.sample()
                            log_prob = distribution.log_prob(value)
                        reused = False
                    elif address not in _metropolis_hastings_trace._samples_all_dict_address:
                        value = distribution.sample()
                        log_prob = distribution.log_prob(value)
                        reused = False
                    else:
                        value = _metropolis_hastings_trace._samples_all_dict_address[address].value
                        reused = True
                        try:  # Takes care of issues such as changed distribution parameters (e.g., batch size) that prevent a rescoring of a reused value under this distribution.
                            log_prob = distribution.log_prob(value)
                        except:
                            value = distribution.sample()
                            log_prob = distribution.log_prob(value)
                            reused = False

        current_sample = Sample(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, control=control, replace=replace, reused=reused)
        _current_trace.add_sample(current_sample)

        if update_previous_sample:
            _current_trace_previous_sample = current_sample

        return current_sample.value


def observe(distribution, observation, address=None):
    if _trace_mode != TraceMode.NONE:
        global _current_trace
        if address is None:
            address_base = extract_address(_current_trace_root_function_name)
        else:
            address_base = address
        instance = _current_trace.last_instance(address_base) + 1
        address = '{}_{}_{}'.format(address_base, distribution.address_suffix, instance)

        log_prob = _observation_importance_exponent * distribution.log_prob(observation)
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            _current_trace.log_importance_weight += log_prob.item()

        current_sample = Sample(distribution=distribution, value=observation, address_base=address_base, address=address, instance=instance, log_prob=log_prob, observed=True)
        _current_trace.add_sample(current_sample)
    return


def begin_trace(func, trace_mode=TraceMode.NONE, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, metropolis_hastings_trace=None, observation_importance_exponent=1.):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _observation_importance_exponent
    _trace_mode = trace_mode
    _inference_engine = inference_engine
    _prior_inflation = prior_inflation
    _observation_importance_exponent = observation_importance_exponent
    if trace_mode != TraceMode.NONE:
        global _current_trace
        global _current_trace_root_function_name
        global _current_trace_inference_network
        global _current_trace_replaced_sample_proposal_distributions
        _current_trace = Trace()
        _current_trace_root_function_name = func.__code__.co_name
        _current_trace_inference_network = inference_network
        _current_trace_replaced_sample_proposal_distributions = {}
        if (_inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK) and (inference_network is None):
            raise ValueError('Cannot run trace with proposals without an inference network')
        if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
            global _metropolis_hastings_trace
            global _metropolis_hastings_site_transition_log_prob
            _metropolis_hastings_trace = metropolis_hastings_trace
            _metropolis_hastings_site_transition_log_prob = None
            if _metropolis_hastings_trace is not None:
                global _metropolis_hastings_site_address
                sample = random.choice(_metropolis_hastings_trace.samples)
                _metropolis_hastings_site_address = sample.address


def end_trace(result):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    _inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
    _prior_inflation = PriorInflation.DISABLED
    if _trace_mode == TraceMode.NONE:
        return None
    else:
        _trace_mode = TraceMode.NONE
        _current_trace.end(result)
        ret = _current_trace
        _current_trace = None
        _current_trace_root_function_name = None
        _current_trace_inference_network = None
        return ret
