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
from torch.autograd import Variable
from termcolor import colored
import math
import datetime

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
        self.input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, self.input_dim)))
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
        self.created = datetime.datetime.now()
        self.modified = datetime.datetime.now()
        self.on_cuda = None
        self.code_version = util.version
        self.pytorch_version = torch.__version__
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
        self.lstm_input_dim = None
        self.smp_emb = None
        self.smp_emb_dim = None
        self.obs_emb = None
        self.obs_emb_dim = None
        self.num_parameters = None
        self.train_loss_best = math.inf
        self.train_loss_worst = -math.inf
        self.valid_loss_best = None
        self.valid_loss_worst = None
        self.valid_loss_initial = None
        self.valid_loss_final = None
        self.valid_history_trace = []
        self.valid_history_loss = []
        self.train_history_trace = []
        self.train_history_loss = []
        self.total_training_time = None
        self.total_iterations = None
        self.total_traces = None
        self.updates = 0
        self.optimizer_state = None

    def get_str(self):
        ret = str(next(enumerate(self.modules()))[1])
        ret = ret + '\nParameters: ' + str(self.num_parameters)
        for p in self.parameters():
            ret = ret + '\n{0} {1}'.format(type(p.data), p.size())
        return ret

    def get_info(self):
        iter_per_sec = self.total_iterations / self.total_training_time.total_seconds()
        traces_per_sec = self.total_traces / self.total_training_time.total_seconds()
        traces_per_iter = self.total_traces / self.total_iterations
        loss_change = self.valid_loss_final - self.valid_loss_initial
        loss_change_per_sec = loss_change / self.total_training_time.total_seconds()
        loss_change_per_iter = loss_change / self.total_iterations
        loss_change_per_trace = loss_change / self.total_traces
        addresses = ' '.join(list(self.one_hot_address.keys()))
        instances = ' '.join(map(str, list(self.one_hot_instance.keys())))
        proposal_types = ' '.join(list(self.one_hot_proposal_type.keys()))
        info = '\n'.join(['Name                  : {0}'.format(self.name),
                          'Created               : {0}'.format(self.created),
                          'Last modified         : {0}'.format(self.modified),
                          'Code version          : {0}'.format(self.code_version),
                          'Cuda                  : {0}'.format(self.on_cuda),
                          colored('Trainable params      : {:,}'.format(self.num_parameters), 'cyan', attrs=['bold']),
                          colored('Total training time   : {0}'.format(util.days_hours_mins_secs(self.total_training_time)), 'yellow', attrs=['bold']),
                          colored('Updates to file       : {:,}'.format(self.updates), 'yellow'),
                          colored('Iterations            : {:,}'.format(self.total_iterations), 'yellow'),
                          colored('Iterations / s        : {:,.2f}'.format(iter_per_sec), 'yellow'),
                          colored('Total training traces : {:,}'.format(self.total_traces), 'yellow', attrs=['bold']),
                          colored('Traces / s            : {:,.2f}'.format(traces_per_sec), 'yellow'),
                          colored('Traces / iteration    : {:,.2f}'.format(traces_per_iter), 'yellow'),
                          colored('Initial loss          : {:+.6e}'.format(self.valid_loss_initial), 'green'),
                          colored('Final loss            : {:+.6e}'.format(self.valid_loss_final), 'green', attrs=['bold']),
                          colored('Loss change / s       : {:+.6e}'.format(loss_change_per_sec), 'green'),
                          colored('Loss change / iter.   : {:+.6e}'.format(loss_change_per_iter), 'green'),
                          colored('Loss change / trace   : {:+.6e}'.format(loss_change_per_trace), 'green'),
                          colored('Validation set size   : {:,}'.format(self.valid_size), 'green'),
                          colored('Observe embedding     : {0}'.format(self.obs_emb), 'cyan'),
                          colored('Observe emb. dim.     : {:,}'.format(self.obs_emb_dim), 'cyan'),
                          colored('Sample embedding      : {0}'.format(self.smp_emb), 'cyan'),
                          colored('Sample emb. dim.      : {:,}'.format(self.smp_emb_dim), 'cyan'),
                          colored('LSTM dim.             : {:,}'.format(self.lstm_dim), 'cyan'),
                          colored('LSTM depth            : {:,}'.format(self.lstm_depth), 'cyan'),
                          colored('Softmax boost         : {0}'.format(self.softmax_boost), 'cyan'),
                          colored('Addresses             : {0}'.format(addresses), 'yellow'),
                          colored('Instances             : {0}'.format(instances), 'yellow'),
                          colored('Proposal types        : {0}'.format(proposal_types), 'yellow')])
        return info

    def polymorph(self, batch=None):
        if batch is None:
            batch = self.valid_batch

        layers_changed = False
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
                    util.log_print(colored('Polymorphing, new layers attached : {0}, {1}'.format(address, instance), 'magenta', attrs=['bold']))
                    layers_changed = True

        if layers_changed:
            self.num_parameters = 0
            for p in self.parameters():
                self.num_parameters += p.nelement()
            util.log_print(colored('Polymorphing, new trainable params: {:,}'.format(self.num_parameters), 'magenta', attrs=['bold']))

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
        self.lstm_input_dim = self.obs_emb_dim + self.smp_emb_dim + self.one_hot_address_dim + self.one_hot_instance_dim + self.one_hot_proposal_type_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_dim, lstm_depth)

    def add_one_hot_address(self, address):
        if not address in self.one_hot_address:
            util.log_print(colored('Polymorphing, new address         : ' + address, 'magenta', attrs=['bold']))
            i = len(self.one_hot_address)
            if i >= self.one_hot_address_dim:
                log_error('one_hot_address overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_address_dim).fill_(0)
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_address[address] = Variable(t, requires_grad=False)

    def add_one_hot_instance(self, instance):
        if not instance in self.one_hot_instance:
            util.log_print(colored('Polymorphing, new instance        : ' + str(instance), 'magenta', attrs=['bold']))
            i = len(self.one_hot_instance)
            if i >= self.one_hot_instance_dim:
                log_error('one_hot_instance overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_instance_dim).fill_(0)
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_instance[instance] = Variable(t, requires_grad=False)

    def add_one_hot_proposal_type(self, proposal_type):
        if not proposal_type in self.one_hot_proposal_type:
            util.log_print(colored('Polymorphing, new proposal type   : ' + proposal_type, 'magenta', attrs=['bold']))
            i = len(self.one_hot_proposal_type)
            if i >= self.one_hot_proposal_type_dim:
                log_error('one_hot_proposal_type overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_proposal_type_dim).fill_(0)
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_proposal_type[proposal_type] = Variable(t, requires_grad=False)

    def valid_loss(self):
        loss = 0
        for sub_batch in self.valid_batch:
            loss += self.loss(sub_batch)
        return loss.data[0] / len(self.valid_batch)

    def loss(self, sub_batch):
        sub_batch_size = len(sub_batch)
        example_observes = sub_batch[0].observes

        if example_observes.dim() == 1:
            obs = torch.cat([sub_batch[b].observes for b in range(sub_batch_size)]).view(sub_batch_size, example_observes.size()[0])
        elif example_observes.dim() == 2:
            obs = torch.cat([sub_batch[b].observes for b in range(sub_batch_size)]).view(sub_batch_size, example_observes.size()[0], example_observes.size()[1])
        else:
            util.log_error('Unsupported observation shape: {0}'.format(example_observes.size()))

        observe_embedding = self.observe_layer(Variable(obs, requires_grad=False))

        example_trace = sub_batch[0]

        lstm_input = []
        for time_step in range(example_trace.length):
            sample = example_trace.samples[time_step]
            address = sample.address
            instance = sample.instance
            proposal_type = sample.proposal_type

            smp = torch.cat([sub_batch[b].samples[time_step].value for b in range(sub_batch_size)]).view(sub_batch_size, sample.value_dim)
            sample_embedding = self.sample_layers[(address, instance)](Variable(smp, requires_grad=False))

            t = []
            for b in range(sub_batch_size):
                t.append(torch.cat([observe_embedding[b],
                               sample_embedding[b],
                               self.one_hot_address[address],
                               self.one_hot_instance[instance],
                               self.one_hot_proposal_type[proposal_type]]))
            t = torch.cat(t).view(sub_batch_size, -1)
            lstm_input.append(t)
        lstm_input = torch.cat(lstm_input).view(example_trace.length, sub_batch_size, -1)

        lstm_output, _ = self.lstm(lstm_input)

        logpdf = 0
        for time_step in range(example_trace.length):
            sample = example_trace.samples[time_step]
            address = sample.address
            instance = sample.instance
            proposal_type = sample.proposal_type

            proposal_input = lstm_output[time_step]
            proposal_output = self.proposal_layers[(address, instance)](proposal_input)

            if proposal_type == 'discreteminmax':
                log_weights = torch.log(proposal_output + util.epsilon)
                for b in range(sub_batch_size):
                    value = sub_batch[b].samples[time_step].value[0]
                    min = sub_batch[b].samples[time_step].proposal_min
                    logpdf += log_weights[b, int(value) - min] # Check this for correctness
            else:
                util.log_error('Unsupported proposal distribution: ' + proposal_type)

        return -logpdf / sub_batch_size
