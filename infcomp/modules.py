#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import infcomp
from infcomp import util
from infcomp.probprog import UniformDiscrete, Normal, Flip, Discrete, Categorical, UniformContinuous
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from termcolor import colored
import math
import datetime
import gc

class Batch(object):
    def __init__(self, traces, sort=True):
        self.batch = traces
        self.length = len(traces)
        self.traces_lengths = []
        self.traces_max_length = 0
        self.observes_max_length = 0
        sb = {}
        for trace in traces:
            if trace.length > self.traces_max_length:
                self.traces_max_length = trace.length
            if trace.observes.size(0) > self.observes_max_length:
                self.observes_max_length = trace.observes.size(0)
            h = hash(str(trace))
            if not h in sb:
                sb[h] = []
            sb[h].append(trace)
        self.sub_batches = []
        for _, t in sb.items():
            self.sub_batches.append(t)
        if sort:
            # Sort the batch in decreasing trace length.
            self.batch = sorted(self.batch, reverse=True, key=lambda t: t.length)
        self.traces_lengths = [t.length for t in self.batch]
    def __getitem__(self, key):
        return self.batch[key]
    def __setitem__(self, key, value):
        self.batch[key] = value
    def cuda(self, device_id=None):
        for trace in self.batch:
            trace.cuda(device_id)
    def cpu(self):
        for trace in self.batch:
            trace.cpu()
    def sort_by_observes_length(self):
        return Batch(sorted(self.batch, reverse=True, key=lambda x:x.observes.nelement()), False)

class ProposalUniformDiscrete(nn.Module):
    def __init__(self, input_dim, output_min, output_dim, softmax_boost=1.0):
        super(ProposalUniformDiscrete, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return F.softmax(self.lin1(x).mul_(self.softmax_boost))

class ProposalNormal(nn.Module):
    def __init__(self, input_dim, prior_mean, prior_std):
        super(ProposalNormal, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        # self.meanMultiplier = nn.Parameter(util.Tensor(1).fill_(1))
        # self.stdMultiplier = nn.Parameter(util.Tensor(1).fill_(1))
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        x = self.lin1(x)
        means = x[:,0].unsqueeze(1)
        stds = x[:,1].unsqueeze(1)
        # means = nn.Tanh()(means)
        # stds = nn.Sigmoid()(stds)
        stds = nn.Softplus()(stds)
        # means = means * self.meanMultiplier.expand_as(means)
        # stds = stds * self.stdMultiplier.expand_as(stds)
        return torch.cat([(means * self.prior_std) + self.prior_mean, stds * self.prior_std], 1)

class ProposalFlip(nn.Module):
    def __init__(self, input_dim, softmax_boost=1.0):
        super(ProposalFlip, self).__init__()
        self.lin1 = nn.Linear(input_dim, 1)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return nn.Sigmoid()(self.lin1(x).mul_(self.softmax_boost))

class ProposalDiscrete(nn.Module):
    def __init__(self, input_dim, output_size, softmax_boost=1.0):
        super(ProposalDiscrete, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_size)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return F.softmax(self.lin1(x).mul_(self.softmax_boost))

class ProposalCategorical(nn.Module):
    def __init__(self, input_dim, output_size, softmax_boost=1.0):
        super(ProposalCategorical, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_size)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return F.softmax(self.lin1(x).mul_(self.softmax_boost))

class ProposalUniformContinuous(nn.Module):
    def __init__(self, input_dim, prior_min, prior_max):
        super(ProposalUniformContinuous, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.prior_min = prior_min
        self.prior_max = prior_max
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        x = self.lin1(x)
        modes = x[:,0].unsqueeze(1)
        ks = x[:,1].unsqueeze(1)
        modes = nn.Sigmoid()(modes)
        ks = nn.Softplus()(ks)
        return torch.cat([(modes * (self.prior_max - self.prior_min) + self.prior_min), ks], 1)

class SampleEmbeddingFC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SampleEmbeddingFC, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        return F.relu(self.lin1(x))

class ObserveEmbeddingFC(nn.Module):
    def __init__(self, input_example_non_batch, output_dim):
        super(ObserveEmbeddingFC, self).__init__()
        self.input_dim = input_example_non_batch.nelement()
        self.lin1 = nn.Linear(self.input_dim, output_dim)
        self.lin2 = nn.Linear(output_dim, output_dim)
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
        init.xavier_uniform(self.lin2.weight, gain=np.sqrt(2.0))
    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, self.input_dim)))
        x = F.relu(self.lin2(x))
        return x

class ObserveEmbeddingLSTM(nn.Module):
    def __init__(self, input_example_non_batch, output_dim):
        super(ObserveEmbeddingLSTM, self).__init__()
        self.input_dim = input_example_non_batch.size(1)
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, self.output_dim, 1, batch_first=True)
    def forward(self, x):
        batch_size = x.size(0)
        # seq_len = x.size(1)
        h0 = Variable(util.Tensor(1, batch_size, self.output_dim).zero_(), requires_grad=False)
        _, (x, _) = self.lstm(x, (h0, h0))
        return x[0]
    def forward_packed(self, x, batch_size):
        h0 = Variable(util.Tensor(1, batch_size, self.output_dim).zero_(), requires_grad=False)
        x, (_, _) = self.lstm(x, (h0, h0))
        return x


class ObserveEmbeddingCNN6(nn.Module):
    def __init__(self, input_example_non_batch, output_dim):
        super(ObserveEmbeddingCNN6, self).__init__()
        if input_example_non_batch.dim() == 2:
            self.input_sample = input_example_non_batch.unsqueeze(0).cpu()
        elif input_example_non_batch.dim() == 3:
            self.input_sample = input_example_non_batch.cpu()
        else:
            util.log_error('Expecting a 3d input_example_non_batch (num_channels x height x width) or a 2d input_example_non_batch (height x width). Received: {0}'.format(input_example_non_batch.size()))
        self.input_channels = self.input_sample.size(0)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(self.input_channels, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
    def configure(self):
        self.cnn_output_dim = self.forward_cnn(self.input_sample.unsqueeze(0)).view(-1).size(0)
        self.lin1 = nn.Linear(self.cnn_output_dim, self.output_dim)
        self.lin2 = nn.Linear(self.output_dim, self.output_dim)
    def forward_cnn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = nn.MaxPool2d(2)(x)
        x = F.relu(self.conv6(x))
        x = nn.MaxPool2d(2)(x)
        return x
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add a channel dimension of 1 after the batch dimension. Temporary. This can be removed once we ensure that we always get 2d images as 3d tensors of form (num_channels x height x width) from the protocol.
        x = self.forward_cnn(x)
        x = x.view(-1, self.cnn_output_dim)
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

        self.model_name = ''
        self.created = datetime.datetime.now()
        self.modified = datetime.datetime.now()
        self.on_cuda = None
        self.cuda_device_id = None
        self.code_version = infcomp.__version__
        self.pytorch_version = torch.__version__
        self.standardize = True
        self.one_hot_address = {}
        self.one_hot_instance = {}
        self.one_hot_distribution = {}
        self.one_hot_address_dim = None
        self.one_hot_instance_dim = None
        self.one_hot_distribution_dim = None
        self.one_hot_address_empty = None
        self.one_hot_instance_empty = None
        self.one_hot_distribution_empty = None
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
        self.total_training_seconds = None
        self.total_iterations = None
        self.total_traces = None
        self.updates = 0
        self.optimizer = None

    def get_structure(self):
        ret = str(next(enumerate(self.modules()))[1])
        for p in self.parameters():
            ret = ret + '\n{0} {1}'.format(type(p.data), p.size())
        return ret

    def get_info(self):
        iter_per_sec = self.total_iterations / self.total_training_seconds
        traces_per_sec = self.total_traces / self.total_training_seconds
        traces_per_iter = self.total_traces / self.total_iterations
        loss_change = self.valid_loss_final - self.valid_loss_initial
        loss_change_per_sec = loss_change / self.total_training_seconds
        loss_change_per_iter = loss_change / self.total_iterations
        loss_change_per_trace = loss_change / self.total_traces
        addresses = ' '.join(list(self.one_hot_address.keys()))
        instances = ' '.join(map(str, list(self.one_hot_instance.keys())))
        distributions = ' '.join(list(self.one_hot_distribution.keys()))
        info = '\n'.join(['Model name            : {0}'.format(self.model_name),
                          'Created               : {0}'.format(self.created),
                          'Modified              : {0}'.format(self.modified),
                          'Code version          : {0}'.format(self.code_version),
                          'Trained on            : CUDA' if self.on_cuda else 'Trained on            : CPU',
                          colored('Trainable params      : {:,}'.format(self.num_parameters), 'cyan', attrs=['bold']),
                          colored('Total training time   : {0}'.format(util.days_hours_mins_secs(self.total_training_seconds)), 'yellow', attrs=['bold']),
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
                          colored('Distributions         : {0}'.format(distributions), 'yellow')])
        return info

    def polymorph(self, batch=None):
        if batch is None:
            batch = self.valid_batch

        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for sample in example_trace.samples:
                address = sample.address
                instance = sample.instance
                distribution = sample.distribution

                # update the artifact's one-hot dictionary as needed
                self.add_one_hot_address(address)
                self.add_one_hot_instance(instance)
                self.add_one_hot_distribution(distribution)

                # update the artifact's sample and proposal layers as needed
                if not (address, instance) in self.sample_layers:
                    if self.smp_emb == 'fc':
                        sample_layer = SampleEmbeddingFC(sample.value.nelement(), self.smp_emb_dim)
                    else:
                        util.log_error('Unsupported sample embedding: ' + self.smp_emb)
                    if isinstance(distribution, UniformDiscrete):
                        proposal_layer = ProposalUniformDiscrete(self.lstm_dim, distribution.prior_min, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self.lstm_dim, distribution.prior_mean, distribution.prior_std)
                    elif isinstance(distribution, Flip):
                        proposal_layer = ProposalFlip(self.lstm_dim, self.softmax_boost)
                    elif isinstance(distribution, Discrete):
                        proposal_layer = ProposalDiscrete(self.lstm_dim, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategorical(self.lstm_dim, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, UniformContinuous):
                        proposal_layer = ProposalUniformContinuous(self.lstm_dim, distribution.prior_min, distribution.prior_max)
                    else:
                        util.log_error('Unsupported distribution: ' + sample.distribution.name())
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
        if self.on_cuda:
            self.cuda(self.cuda_device_id)
        return layers_changed

    def set_sample_embedding(self, smp_emb, smp_emb_dim):
        self.smp_emb = smp_emb
        self.smp_emb_dim = smp_emb_dim

    def set_observe_embedding(self, example_observes, obs_emb, obs_emb_dim):
        self.obs_emb = obs_emb
        self.obs_emb_dim = obs_emb_dim
        if obs_emb == 'fc':
            observe_layer = ObserveEmbeddingFC(Variable(example_observes), obs_emb_dim)
        elif obs_emb == 'cnn6':
            observe_layer = ObserveEmbeddingCNN6(Variable(example_observes), obs_emb_dim)
            observe_layer.configure()
        elif obs_emb == 'lstm':
            observe_layer = ObserveEmbeddingLSTM(Variable(example_observes), obs_emb_dim)
        else:
            util.log_error('Unsupported observation embedding: ' + obs_emb)

        self.observe_layer = observe_layer

    def set_lstm(self, lstm_dim, lstm_depth, dropout):
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.lstm_input_dim = self.obs_emb_dim + self.smp_emb_dim + 2 * (self.one_hot_address_dim + self.one_hot_instance_dim + self.one_hot_distribution_dim)
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_dim, lstm_depth, dropout=dropout)

    def set_one_hot_dims(self, one_hot_address_dim, one_hot_instance_dim, one_hot_distribution_dim):
        self.one_hot_address_dim =one_hot_address_dim
        self.one_hot_instance_dim = one_hot_instance_dim
        self.one_hot_distribution_dim = one_hot_distribution_dim
        self.one_hot_address_empty = Variable(util.Tensor(self.one_hot_address_dim).zero_(), requires_grad=False)
        self.one_hot_instance_empty = Variable(util.Tensor(self.one_hot_instance_dim).zero_(), requires_grad=False)
        self.one_hot_distribution_empty = Variable(util.Tensor(self.one_hot_distribution_dim).zero_(), requires_grad=False)

    def add_one_hot_address(self, address):
        if not address in self.one_hot_address:
            util.log_print(colored('Polymorphing, new address         : ' + address, 'magenta', attrs=['bold']))
            i = len(self.one_hot_address)
            if i >= self.one_hot_address_dim:
                log_error('one_hot_address overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_address_dim).zero_()
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_address[address] = Variable(t, requires_grad=False)

    def add_one_hot_instance(self, instance):
        if not instance in self.one_hot_instance:
            util.log_print(colored('Polymorphing, new instance        : ' + str(instance), 'magenta', attrs=['bold']))
            i = len(self.one_hot_instance)
            if i >= self.one_hot_instance_dim:
                log_error('one_hot_instance overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_instance_dim).zero_()
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_instance[instance] = Variable(t, requires_grad=False)

    def add_one_hot_distribution(self, distribution):
        distribution_name = distribution.name()
        if not distribution_name in self.one_hot_distribution:
            util.log_print(colored('Polymorphing, new distribution    : ' + distribution_name, 'magenta', attrs=['bold']))
            i = len(self.one_hot_distribution)
            if i >= self.one_hot_distribution_dim:
                util.log_error('one_hot_distribution overflow: {0}'.format(i))
            t = util.Tensor(self.one_hot_distribution_dim).zero_()
            t.narrow(0, i, 1).fill_(1)
            self.one_hot_distribution[distribution_name] = Variable(t, requires_grad=False)

    def move_to_cuda(self, device_id=None):
        self.on_cuda = True
        self.cuda_device_id = device_id
        self.cuda(device_id)
        self.one_hot_address_empty = self.one_hot_address_empty.cuda(device_id)
        self.one_hot_instance_empty = self.one_hot_instance_empty.cuda(device_id)
        self.one_hot_distribution_empty = self.one_hot_distribution_empty.cuda(device_id)
        for k, t in self.one_hot_address.items():
            self.one_hot_address[k] = t.cuda(device_id)
        for k, t in self.one_hot_instance.items():
            self.one_hot_instance[k] = t.cuda(device_id)
        for k, t in self.one_hot_distribution.items():
            self.one_hot_distribution[k] = t.cuda(device_id)
        self.valid_batch.cuda(device_id)

    def move_to_cpu(self):
        self.on_cuda = False
        self.cpu()
        self.one_hot_address_empty = self.one_hot_address_empty.cpu()
        self.one_hot_instance_empty = self.one_hot_instance_empty.cpu()
        self.one_hot_distribution_empty = self.one_hot_distribution_empty.cpu()
        for k, t in self.one_hot_address.items():
            self.one_hot_address[k] = t.cpu()
        for k, t in self.one_hot_instance.items():
            self.one_hot_instance[k] = t.cpu()
        for k, t in self.one_hot_distribution.items():
            self.one_hot_distribution[k] = t.cpu()
        self.valid_batch.cpu()

    def valid_loss(self, data_parallel=False):
        return self.loss(self.valid_batch, data_parallel, False).data[0]

    def loss(self, batch, data_parallel=False, volatile=False):
        gc.collect()

        example_observes = batch[0].observes
        if isinstance(self.observe_layer, ObserveEmbeddingLSTM):
            if example_observes.dim() != 2:
                util.log_error('RNN observation embedding requires an observation shape of (T x F), where T is sequence length and F is feature length. Received observation with shape: {0}'.format(example_observes.size()))

            # We need to sort observes in the batch in decreasing length. This is a requirement for batching variable length sequences through RNNs.
            batch_sorted_by_observes = batch.sort_by_observes_length()
            observes_lengths = [t.observes.size(0) for t in batch_sorted_by_observes]
            observes_feature_length = batch[0].observes.size(1)
            obs = util.Tensor(batch.length, batch.observes_max_length, observes_feature_length).zero_()
            for b in range(batch.length):
                o = batch_sorted_by_observes[b].observes
                for t in range(o.size(0)):
                    obs[b, t] = o[t]
            obs_var = Variable(obs, requires_grad=False, volatile=volatile)
            obs_var = torch.nn.utils.rnn.pack_padded_sequence(obs_var, observes_lengths, batch_first=True)
            obs_emb = self.observe_layer.forward_packed(obs_var, batch.length)
            obs_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(obs_emb, batch_first=True)

            for b in range(batch.length):
                seq_len = batch_sorted_by_observes[b].observes.size(0)
                batch_sorted_by_observes[b].observes_embedding = obs_emb[b, seq_len - 1]
        else:
            obs = torch.cat([batch[b].observes for b in range(batch.length)]);
            if example_observes.dim() == 1:
                obs = obs.view(batch.length, example_observes.size()[0])
            elif example_observes.dim() == 2:
                obs = obs.view(batch.length, example_observes.size()[0], example_observes.size()[1])
            elif example_observes.dim() == 3:
                obs = obs.view(batch.length, example_observes.size()[0], example_observes.size()[1], example_observes.size()[2])
            else:
                util.log_error('Unsupported observation dimensions: {0}'.format(example_observes.size()))

            obs_var = Variable(obs, requires_grad=False, volatile=volatile)
            if data_parallel and self.on_cuda:
                obs_emb = torch.nn.DataParallel(self.observe_layer)(obs_var)
            else:
                obs_emb = self.observe_layer(obs_var)

            for b in range(batch.length):
                batch[b].observes_embedding = obs_emb[b]

        for sub_batch in batch.sub_batches:
            sub_batch_size = len(sub_batch)
            example_trace = sub_batch[0]
            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address
                current_instance = current_sample.instance
                current_distribution = current_sample.distribution

                current_one_hot_address = self.one_hot_address[current_address]
                current_one_hot_instance = self.one_hot_instance[current_instance]
                current_one_hot_distribution = self.one_hot_distribution[current_distribution.name()]

                if time_step == 0:
                    prev_sample_embedding = Variable(util.Tensor(sub_batch_size, self.smp_emb_dim).zero_(), requires_grad=False, volatile=volatile)

                    prev_one_hot_address = self.one_hot_address_empty
                    prev_one_hot_instance = self.one_hot_instance_empty
                    prev_one_hot_distribution = self.one_hot_distribution_empty
                else:
                    prev_sample = example_trace.samples[time_step - 1]
                    prev_address = prev_sample.address
                    prev_instance = prev_sample.instance
                    prev_distribution = prev_sample.distribution
                    smp = torch.cat([sub_batch[b].samples[time_step - 1].value for b in range(sub_batch_size)]).view(sub_batch_size, prev_sample.value.nelement())

                    smp_var = Variable(smp, requires_grad=False, volatile=volatile)
                    sample_layer = self.sample_layers[(prev_address, prev_instance)]
                    # if data_parallel and self.on_cuda:
                    #     prev_sample_embedding = torch.nn.DataParallel(sample_layer)(smp_var)
                    # else:
                    prev_sample_embedding = sample_layer(smp_var)

                    prev_one_hot_address = self.one_hot_address[prev_address]
                    prev_one_hot_instance = self.one_hot_instance[prev_instance]
                    prev_one_hot_distribution = self.one_hot_distribution[prev_distribution.name()]

                for b in range(sub_batch_size):
                    t = torch.cat([sub_batch[b].observes_embedding,
                                   prev_sample_embedding[b],
                                   prev_one_hot_address,
                                   prev_one_hot_instance,
                                   prev_one_hot_distribution,
                                   current_one_hot_address,
                                   current_one_hot_instance,
                                   current_one_hot_distribution])
                    sub_batch[b].samples[time_step].lstm_input = t


        lstm_input = []
        for time_step in range(batch.traces_max_length):
            t = []
            for b in range(batch.length):
                trace = batch[b]
                if time_step < trace.length:
                    t.append(trace.samples[time_step].lstm_input)
                else:
                    t.append(Variable(util.Tensor(self.lstm_input_dim).zero_(), volatile=volatile))
            t = torch.cat(t).view(batch.length, -1)
            lstm_input.append(t)
        lstm_input = torch.cat(lstm_input).view(batch.traces_max_length, batch.length, -1)
        lstm_input = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, batch.traces_lengths)

        h0 = Variable(util.Tensor(self.lstm_depth, batch.length, self.lstm_dim).zero_(), requires_grad=False, volatile=volatile)
        if data_parallel and self.on_cuda:
            lstm_output, _ = torch.nn.DataParallel(self.lstm, dim=1)(lstm_input, (h0, h0))
        else:
            lstm_output, _ = self.lstm(lstm_input, (h0, h0))
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)

        for b in range(batch.length):
            trace = batch[b]
            for time_step in range(trace.length):
                trace.samples[time_step].lstm_output = lstm_output[time_step, b]


        logpdf = 0
        for sub_batch in batch.sub_batches:
            sub_batch_size = len(sub_batch)
            example_trace = sub_batch[0]

            for time_step in range(example_trace.length):
                current_sample = example_trace.samples[time_step]
                current_address = current_sample.address
                current_instance = current_sample.instance
                current_distribution = current_sample.distribution

                p = []
                for b in range(sub_batch_size):
                    p.append(sub_batch[b].samples[time_step].lstm_output)
                p = torch.cat(p).view(sub_batch_size, -1)

                proposal_input = p
                proposal_layer = self.proposal_layers[(current_address, current_instance)]
                # if data_parallel and self.on_cuda:
                #     proposal_output = torch.nn.DataParallel(proposal_layer)(proposal_input)
                # else:
                proposal_output = proposal_layer(proposal_input)

                if isinstance(current_distribution, UniformDiscrete):
                    log_weights = torch.log(proposal_output + util.epsilon)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        min = sub_batch[b].samples[time_step].distribution.prior_min
                        logpdf += log_weights[b, int(value) - min] # Should we average this over dimensions? See http://pytorch.org/docs/nn.html#torch.nn.KLDivLoss
                elif isinstance(current_distribution, Normal):
                    means = proposal_output[:, 0]
                    stds = proposal_output[:, 1]

                    # prior_means = util.Tensor([sub_batch[b].samples[time_step].distribution.prior_mean for b in range(sub_batch_size)])
                    # prior_stds =  util.Tensor([sub_batch[b].samples[time_step].distribution.prior_std for b in range(sub_batch_size)])

                    two_std_squares = 2 * stds * stds + util.epsilon
                    two_pi_std_squares = math.pi * two_std_squares
                    half_log_two_pi_std_squares = 0.5 * torch.log(two_pi_std_squares + util.epsilon)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        mean = means[b]
                        two_std_square = two_std_squares[b]
                        half_log_two_pi_std_square = half_log_two_pi_std_squares[b]
                        logpdf -= half_log_two_pi_std_square + ((value - mean)**2) / two_std_square
                elif isinstance(current_distribution, Flip):
                    log_probabilities = torch.log(proposal_output + util.epsilon)
                    log_one_minus_probabilities = torch.log(1 - proposal_output + util.epsilon)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        if value > 0:
                            logpdf += log_probabilities[b]
                        else:
                            logpdf += log_one_minus_probabilities[b]
                elif isinstance(current_distribution, Discrete):
                    log_weights = torch.log(proposal_output + util.epsilon)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        logpdf += log_weights[b, int(value)]
                elif isinstance(current_distribution, Categorical):
                    log_weights = torch.log(proposal_output + util.epsilon)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        logpdf += log_weights[b, int(value)]
                elif isinstance(current_distribution, UniformContinuous):
                    normalized_modes = proposal_output[:, 0]
                    normalized_ks = proposal_output[:, 1] + 2
                    alphas = normalized_modes * (normalized_ks - 2) + 1
                    betas = (1 - normalized_modes) * (normalized_ks - 2) + 1
                    beta_funs = util.beta(alphas, betas)
                    for b in range(sub_batch_size):
                        value = sub_batch[b].samples[time_step].value[0]
                        prior_min = sub_batch[b].samples[time_step].distribution.prior_min
                        prior_max = sub_batch[b].samples[time_step].distribution.prior_max
                        normalized_value = (value - prior_min) / (prior_max - prior_min)
                        alpha = alphas[b]
                        beta = betas[b]
                        beta_fun = beta_funs[b]
                        logpdf += (alpha - 1) * np.log(normalized_value + util.epsilon) + (beta - 1) * np.log(1 - normalized_value + util.epsilon) - torch.log(beta_fun) - np.log(prior_max - prior_min)
                else:
                    util.log_error('Unsupported distribution: ' + current_distribution.name())

        return -logpdf / batch.length
