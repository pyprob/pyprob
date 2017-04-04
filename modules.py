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

class Sample_embedding_fc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Sample_embedding_fc, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return F.relu(self.lin1(x))

class Observe_embedding_fc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Observe_embedding_fc, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

class Artifact(nn.Module):
    def __init__(self):
        super(Artifact, self).__init__()

        self.sample_layers = {}
        self.proposal_layers = {}
        self.observe_layer = None
        self.lstm = None

        self.name = ''
        self.code_version = util.version
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
                    if self.smp_emb == 'fc':
                        sample_layer = Sample_embedding_fc(sample.value_dim, self.smp_emb_dim)
                    else:
                        util.log_error('Unsupported sample embedding: ' + self.smp_emb)
                    if sample.proposal_type == 'discreteminmax':
                        proposal_layer = Proposal_discreteminmax(self.lstm_dim, sample.proposal_min, sample.proposal_max, self.softmax_boost)
                    else:
                        util.log_error('Unsupported proposal distribution: ' + sample.proposal_type)
                    self.sample_layers[(address, instance)] = sample_layer
                    self.proposal_layers[(address, instance)] = proposal_layer
                    self.add_module('sample_layer({0}, {1})'.format(address, instance), sample_layer)
                    self.add_module('proposal_layer({0}, {1})'.format(address, instance), proposal_layer)
                    util.log_print(colored('Polymorphing, new layers attached: {0}, {1}'.format(address, instance), 'magenta', attrs=['bold']))

    def set_sample_embedding(self, smp_emb, smp_emb_dim):
        self.smp_emb = smp_emb
        self.smp_emb_dim = smp_emb_dim

    def set_observe_embedding(self, example_observes, obs_emb, obs_emb_dim):
        self.obs_emb = obs_emb
        self.obs_emb_dim = obs_emb_dim
        if obs_emb == 'fc':
            observe_layer = Observe_embedding_fc(example_observes.nelement(), obs_emb_dim)
        self.observe_layer = observe_layer

    def set_lstm(self, lstm_dim, lstm_depth):
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        lstm_input_dim = self.obs_emb_dim + self.smp_emb_dim + self.one_hot_address_dim + self.one_hot_instance_dim + self.one_hot_proposal_type_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_dim, lstm_depth)

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
