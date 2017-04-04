#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored

class Sample(object):
    def __init__(self, address, instance, value, proposal_type):
        self.address = address
        self.instance = instance
        self.value = value
        self.value_dim = value.nelement()
        self.proposal_type = proposal_type
        self.proposal_min = None
        self.proposal_max = None

    def __repr__(self):
        return 'Sample({0}, {1}, {2}, {3})'.format(self.address, self.instance, self.value.size(), self.proposal_type)
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


class Proposal_discreteminmax(nn.Module):
    def __init__(self, input_dim, output_min, output_max, softmax_boost=1.0):
        super(Proposal_discreteminmax, self).__init__()
        output_dim = output_max - output_min
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.softmax_boost = softmax_boost
    def forward(self, x):
        return F.softmax(F.relu(self.lin1(x)).mul_(self.softmax_boost))


class Artifact(nn.Module):
    def __init__(self):
        super(Artifact, self).__init__()

        self.sample_layers = {}
        self.proposal_layers = {}

        self.name = ''
        self.code_version = util.version
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
        self.lstm_dim = None
        self.lstm_depth = None
        self.smp_emb = None
        self.smp_emb_dim = None
        self.obs_emb = None
        self.obs_emb_dim = None

    def polymorph(self, batch=None):
        if batch is None:
            batch = self.valid_batch

        for sub_batch in batch:
            example_trace = sub_batch[0]
            for sample in example_trace.samples:
                address = sample.address
                instance = sample.instance
                proposal_type = sample.proposal_type

                # update the artifact's one-hot dictionary as needed
                self.add_one_hot_address(address)
                self.add_one_hot_instance(instance)
                self.add_one_hot_proposal_type(proposal_type)

                # update the artifact's sample and proposal layers as needed
                if not (address, instance) in self.sample_layers:
                    self.sample_layers[(address, instance)] = nn.Linear(sample.value_dim, self.smp_emb_dim)
                    if sample.proposal_type == 'discreteminmax':
                        self.proposal_layers[(address, instance)] = Proposal_discreteminmax(self.lstm_dim, sample.proposal_min, sample.proposal_max, self.softmax_boost)
                    else:
                        util.log_error('Unsupported proposal distribution type: ' + sample.proposal_type)
                    util.log_print(colored('Polymorphing, new layers attached: {0}, {1}'.format(address, instance), 'magenta', attrs=['bold']))

    def add_one_hot_address(self, address):
        if not address in self.one_hot_address:
            util.log_print(colored('Polymorphing, new address        : ' + address, 'magenta', attrs=['bold']))
            i = len(self.one_hot_address)
            if i >= self.one_hot_address_dim:
                log_error('one_hot_address overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_address_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_address[address] = t

    def add_one_hot_instance(self, instance):
        if not instance in self.one_hot_instance:
            util.log_print(colored('Polymorphing, new instance       : ' + str(instance), 'magenta', attrs=['bold']))
            i = len(self.one_hot_instance)
            if i >= self.one_hot_instance_dim:
                log_error('one_hot_instance overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_instance_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_instance[instance] = t

    def add_one_hot_proposal_type(self, proposal_type):
        if not proposal_type in self.one_hot_proposal_type:
            util.log_print(colored('Polymorphing, new proposal type  : ' + proposal_type, 'magenta', attrs=['bold']))
            i = len(self.one_hot_proposal_type)
            if i >= self.one_hot_proposal_type_dim:
                log_error('one_hot_proposal_type overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_proposal_type_dim).fill_(0).narrow(0, i, 1).fill_(1)
            self.one_hot_proposal_type[proposal_type] = t
