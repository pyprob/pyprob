import sys
import enum
from .trace import Sample, Trace

class State(enum.Enum):
    NONE = 0
    BUILD_TRACE = 1

state = State.NONE
current_trace = None
current_function_name = None

def extract_address():
    #tb = traceback.extract_stack()
    # print()
    # for t in tb:
    #     print(t[0], t[1], t[2], t[3])
    #frame = tb[-3]
    # return '{0}/{1}/{2}'.format(frame[1], frame[2], frame[3])
    #return '{0}/{1}'.format(frame[1], frame[2])
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
        if n == current_function_name: break
        frame = frame.f_back
    return "{}/{}".format(ip, '.'.join(reversed(names)))


def _extract_target_of_assignment():
    import opcode
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

def sample(distribution):
    value = distribution.sample()
    if state == State.BUILD_TRACE:
        global current_trace
        address = extract_address()
        current_sample = Sample(address, distribution, value)
        current_trace.add_sample(current_sample)
        # current_trace.add_log_prob()
    return value

def observe(distribution, value):
    if state == State.BUILD_TRACE:
        global current_trace
        current_trace.add_observe(value)
        current_trace.add_log_prob(distribution.log_prob(value))
    return

def begin_trace(func):
    global state
    global current_trace
    global current_function_name
    state = State.BUILD_TRACE
    current_trace = Trace()
    current_function_name = func.__code__.co_name

def end_trace():
    global state
    global current_trace
    global current_function_name
    state = State.NONE
    ret = current_trace
    current_trace = None
    current_function_name = None
    return ret
