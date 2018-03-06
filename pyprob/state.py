import sys
import opcode
import enum

from .trace import Sample, Trace


class TraceState(enum.Enum):
    NONE = 0  # No trace recording, forward sample
    RECORD = 1  # Record traces
    RECORD_IMPORTANCE = 2  # Record traces, importance sampling with prior
    RECORD_TRAIN_INFERENCE_NETWORK = 3  # Record traces, training data generation for inference network, interpret 'observe' as 'sample' (inference compilation training)
    RECORD_USE_INFERENCE_NETWORK = 4  # Record traces, importance sampling with proposals from inference network (inference compilation inference)


_trace_state = TraceState.NONE
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None


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
    if var_name is not None:
        names.append(var_name)
    while frame is not None:
        n = frame.f_code.co_name
        if n.startswith('<'): break
        names.append(n)
        if n == root_function_name: break
        frame = frame.f_back
    return "{}/{}".format(ip, '.'.join(reversed(names)))


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
    value = distribution.sample()
    if _trace_state != TraceState.NONE:
        global _current_trace
        if address is None:
            address = extract_address(_current_trace_root_function_name)

        if control:
            if _trace_state == TraceState.RECORD_IMPORTANCE:
                # The log_prob of samples are zero for regular importance sampling (no learned proposals) as it cancels out
                current_sample = Sample(address, distribution, value, log_prob=0, controlled=True)
            else:
                current_sample = Sample(address, distribution, value, log_prob=distribution.log_prob(value), controlled=True)

            if _trace_state == TraceState.RECORD_USE_INFERENCE_NETWORK:
                previous_sample = None
                if _current_trace.length > 0:
                    previous_sample = _current_trace.samples[-1]
                _current_trace_inference_network.eval()
                proposal_distribution = _current_trace_inference_network.forward_one_time_step(previous_sample, current_sample)
                value = proposal_distribution.sample()[0]
                current_sample = Sample(address, distribution, value, log_prob=distribution.log_prob(value) - proposal_distribution.log_prob(value), controlled=True)
        else:
            current_sample = Sample(address, distribution, value, log_prob=0, controlled=False)
        _current_trace.add_sample(current_sample, replace)
    return value


def observe(distribution, observation, address=None):
    if _trace_state != TraceState.NONE:
        global _current_trace
        if address is None:
            address = extract_address(_current_trace_root_function_name)
        if _trace_state == TraceState.RECORD_TRAIN_INFERENCE_NETWORK:
            observation = distribution.sample()
        current_sample = Sample(address, distribution, observation, log_prob=distribution.log_prob(observation), controlled=False, observed=True)
        _current_trace.add_sample(current_sample)
    return


def begin_trace(func, trace_state=TraceState.RECORD, inference_network=None):
    global _trace_state
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    _trace_state = trace_state
    _current_trace = Trace()
    _current_trace_root_function_name = func.__code__.co_name
    _current_trace_inference_network = inference_network
    if (trace_state == TraceState.RECORD_USE_INFERENCE_NETWORK) and (inference_network is None):
        raise ValueError('Cannot run trace with proposals without an inference network')


def end_trace(result):
    global _trace_state
    global _current_trace
    global _current_trace_root_function_name
    global _current_trace_inference_network
    _trace_state = TraceState.NONE
    _current_trace.end(result)
    ret = _current_trace
    _current_trace = None
    _current_trace_root_function_name = None
    _current_trace_inference_network = None
    return ret
