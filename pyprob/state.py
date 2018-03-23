import sys
import opcode
import enum
import pdb

from .trace import Sample, Trace
from . import TraceMode

_trace_mode = TraceMode.NONE
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_previous_sample = None
_current_trace_replaced_sample_proposal_distributions = {}
_current_trace_reassignments = {}
_continue_trace_at = ''
_use_previous_trace_value = False

def extract_address(root_function_name, sampled=False):
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
    if var_name is not None:
        names.append(var_name)
    while frame is not None:
        n = frame.f_code.co_name
        if n.startswith('<'): break
        names.append(n)
        if n == root_function_name: break
        frame = frame.f_back
    iter_id = 0
    byte_address = "{}/{}".format(ip, '.'.join(reversed(names)))
    global _current_trace_reassignments
    if (sampled == True) and (byte_address in _current_trace_reassignments):
        iter_id = _current_trace_reassignments[byte_address] + 1
    _current_trace_reassignments[byte_address] = iter_id
    byte_address += str(iter_id)
    
    return byte_address

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
    else:
        return None


def sample(distribution, control=True, replace=False, address=None):
    global _use_previous_trace_value
    global _previous_trace_values
    global _current_trace
    global _continue_trace_at 
    if address is None:
        address = extract_address(_current_trace_root_function_name, sampled=True)

    if _continue_trace_at == address:
        _current_trace._resampled = _previous_trace_values[address]
        _use_previous_trace_value = False
    if _use_previous_trace_value == True:
        current_sample = _previous_trace_values[address]
        _current_trace.add_sample(current_sample, replace, fresh=False)
        return current_sample.value

    value = distribution.sample()
    if _trace_mode != TraceMode.NONE:
        if control:
            if _trace_mode == TraceMode.RECORD_IMPORTANCE:
                # The log_prob of samples are zero for regular importance sampling (no learned proposals) as it cancels out
                #  log_prob = 0
                log_prob = distribution.log_prob(value)
            else:
                log_prob = distribution.log_prob(value)

            current_sample = Sample(address, distribution, value, log_prob=log_prob, controlled=True)
            if _trace_mode == TraceMode.RECORD_USE_INFERENCE_NETWORK:
                global _current_trace_previous_sample
                global _current_trace_replaced_sample_proposal_distributions
                _current_trace_inference_network.eval()
                if replace:
                    if address not in _current_trace_replaced_sample_proposal_distributions:
                        _current_trace_replaced_sample_proposal_distributions[address] = _current_trace_inference_network.forward_one_time_step(_current_trace_previous_sample, current_sample)
                        _current_trace_previous_sample = current_sample
                    proposal_distribution = _current_trace_replaced_sample_proposal_distributions[address]
                else:
                    proposal_distribution = _current_trace_inference_network.forward_one_time_step(_current_trace_previous_sample, current_sample)
                value = proposal_distribution.sample()[0]
                current_sample = Sample(address, distribution, value, log_prob=distribution.log_prob(value) - proposal_distribution.log_prob(value), controlled=True)

                if not replace:
                    _current_trace_previous_sample = current_sample
        else:
            current_sample = Sample(address, distribution, value, log_prob=distribution.log_prob(value), controlled=False)
        _current_trace.add_sample(current_sample, replace)
    return value


def observe(distribution, observation, address=None):
    if _trace_mode != TraceMode.NONE:
        global _current_trace
        if address is None:
            address = extract_address(_current_trace_root_function_name)
        if _trace_mode == TraceMode.RECORD_TRAIN_INFERENCE_NETWORK:
            observation = distribution.sample()
        current_sample = Sample(address, distribution, observation, log_prob=distribution.log_prob(observation), controlled=False, observed=True)
        _current_trace.add_sample(current_sample)
    return

def dictify_previous_trace(previous_trace):
    d = {}
    for sample in (previous_trace.samples+previous_trace.samples_uncontrolled+previous_trace.samples_observed):
        d[sample.address] = sample
    return d

def begin_trace(func, trace_mode=TraceMode.RECORD, inference_network=None, continuation_address=None, previous_trace=None):
    global _trace_mode
    global _current_trace
    global _previous_trace_values
    global _current_trace_root_function_name
    global _current_trace_inference_network
    global _continue_trace_at
    global _use_previous_trace_value
    global _current_trace_reassignments 
    _use_previous_trace_value = False
    if continuation_address is not None:
        _use_previous_trace_value = True
        _previous_trace_values = dictify_previous_trace(previous_trace)
    _trace_mode = trace_mode
    _current_trace = Trace()
    _current_trace_root_function_name = func.__code__.co_name
    _current_trace_inference_network = inference_network
    _continue_trace_at = continuation_address
    _current_trace_reassignments = {}
    if (trace_mode == TraceMode.RECORD_USE_INFERENCE_NETWORK) and (inference_network is None):
        raise ValueError('Cannot run trace with proposals without an inference network')


def end_trace(result):
    global _trace_mode
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    global _continue_trace_at
    global _use_previous_trace_value
    global _current_trace_reassignments 
    _trace_mode = TraceMode.NONE
    _current_trace.end(result)
    ret = _current_trace
    _current_trace = None
    _current_trace_root_function_name = None
    _current_trace_inference_network = None
    _continue_trace_at = ''
    _use_previous_trace_value = False
    _current_trace_reassignments = {}
    return ret
