#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import torch
import time
import datetime
import logging
import sys
import re
import zmq
import msgpack
from termcolor import colored

version = '0.9.1'
epsilon = 1e-5


class Sample(object):
    def __init__(self, address, instance, value, proposal_type):
        self.address = address
        self.instance = instance
        self.value = value
        self.proposal_type = proposal_type
        self.proposal_extra_params_min = None
        self.proposal_extra_params_max = None

    def __repr__(self):
        return 'Sample({0}, {1}, {2}, {3})'.format(self.address, self.instance, self.proposal_type, self.value)
    __str__ = __repr__


class Trace(object):
    def __init__(self):
        self.observes = None
        self.samples = []
        self.length = None
        self.hash = None

    def __repr__(self):
        return 'Trace(length:{0}; samples:{1}; observes:{2}'.format(self.length, '|'.join(['{0}({1})'.format(sample.address, sample.instance) for sample in  self.samples]), self.observes.size()) + ')'
    __str__ = __repr__

    def set_observes(self, o):
        self.observes = o

    def add_sample(self, s):
        self.samples.append(s)
        self.length = len(self.samples)


class Artifact(object):
    def __init__(self):
        self.name = ''
        self.code_version = version
        self.cuda = False
        self.standardize = True
        self.one_hot_address = {}
        self.one_hot_instance = {}
        self.one_hot_proposal_type = {}
        self.one_hot_address_dim = None
        self.one_hot_instance_dim = None
        self.one_hot_proposal_type_dim = None
        self.valid_size = None
        self.valid_batch = None

    def add_one_hot_address(self, address):
        if not address in self.one_hot_address:
            log_print(colored('Polymorphing, new address      : ' + address, 'magenta', attrs=['bold']))
            i = len(self.one_hot_address)
            if i >= self.one_hot_address_dim:
                log_error('one_hot_address overflow: {0}'.format(i))
            t = Tensor(self.one_hot_address_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_address[address] = t

    def add_one_hot_instance(self, instance):
        if not instance in self.one_hot_instance:
            log_print(colored('Polymorphing, new instance     : ' + str(instance), 'magenta', attrs=['bold']))
            i = len(self.one_hot_instance)
            if i >= self.one_hot_instance_dim:
                log_error('one_hot_instance overflow: {0}'.format(i))
            t = Tensor(self.one_hot_instance_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_instance[instance] = t

    def add_one_hot_proposal_type(self, proposal_type):
        if not proposal_type in self.one_hot_proposal_type:
            log_print(colored('Polymorphing, new proposal type: ' + proposal_type, 'magenta', attrs=['bold']))
            i = len(self.one_hot_proposal_type)
            if i >= self.one_hot_proposal_type_dim:
                log_error('one_hot_proposal_type overflow: {0}'.format(i))
            t = Tensor(self.one_hot_proposal_type_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_proposal_type[proposal_type] = t

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')

def init(opt):
    global Tensor
    torch.manual_seed(opt.seed)
    if opt.cuda:
        if not torch.cuda.is_available():
            util.log_print(colored('Error: CUDA not available', 'red'))
            quit()
        torch.cuda.manual_seed(opt.seed)
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

def init_logger(file_name):
    global logger
    logger = logging.getLogger()
    logger_file_handler = logging.FileHandler(file_name)
    logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logger_file_handler)
    logger.setLevel(logging.INFO)

def log_print(line):
    print(line)
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    logger.info(ansi_escape.sub('', line))

def log_error(line):
    print(colored('Error: ' + line, 'red'))
    logger.error('Error: ' + line)
    quit()

def log_warning(line):
    print(colored('Warning: ' + line, 'yellow'))
    logger.warning('Warning: ' + line)

def zmq_send_request(server_address, request):
    global context
    global socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(server_address)
    socket.send(msgpack.packb(request))

def zmq_receive_reply():
    ret = msgpack.unpackb(socket.recv(), encoding='utf-8')
    socket.close()
    context.term()
    return ret

def standardize(t):
    mean = torch.mean(t)
    sd = torch.std(t)
    t.add_(-mean)
    t.div_(sd + epsilon)
    return t
