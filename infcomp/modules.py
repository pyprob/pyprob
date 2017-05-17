#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- May 2017
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
import sys
import random

class Batch(object):
    def __init__(self, traces, sort=True):
        self.batch = traces
        self.length = len(traces)
        self.traces_lengths = []
        self.traces_max_length = 0
        self.observes_max_length = 0
        sb = {}
        for trace in traces:
            if trace.length is None:
                util.log_error('Batch: Received a trace of length zero.')
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
    def __init__(self, input_dim, output_dim, softmax_boost=1.0):
        super(ProposalUniformDiscrete, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples=None):
        return True, F.softmax(self.lin1(x).mul_(self.softmax_boost))
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x)
        batch_size = len(samples)
        log_weights = torch.log(proposal_output + util.epsilon)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            min = samples[b].distribution.prior_min
            l += log_weights[b, int(value) - min] # Should we average this over dimensions? See http://pytorch.org/docs/nn.html#torch.nn.KLDivLoss
        return l

class ProposalNormal(nn.Module):
    def __init__(self, input_dim):
        super(ProposalNormal, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples):
        x = self.lin1(x)
        means = x[:,0].unsqueeze(1)
        stds = x[:,1].unsqueeze(1)
        stds = nn.Softplus()(stds)
        prior_means = Variable(util.Tensor([s.distribution.prior_mean for s in samples]), requires_grad=False)
        prior_stds = Variable(util.Tensor([s.distribution.prior_std for s in samples]), requires_grad=False)
        return True, torch.cat([(means * prior_stds) + prior_means, stds * prior_stds], 1)
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x, samples)
        batch_size = len(samples)
        means = proposal_output[:, 0]
        stds = proposal_output[:, 1]
        two_std_squares = 2 * stds * stds + util.epsilon
        two_pi_std_squares = math.pi * two_std_squares
        half_log_two_pi_std_squares = 0.5 * torch.log(two_pi_std_squares + util.epsilon)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            mean = means[b]
            two_std_square = two_std_squares[b]
            half_log_two_pi_std_square = half_log_two_pi_std_squares[b]
            l -= half_log_two_pi_std_square + ((value - mean)**2) / two_std_square
        return l

class ProposalFlip(nn.Module):
    def __init__(self, input_dim, softmax_boost=1.0):
        super(ProposalFlip, self).__init__()
        self.lin1 = nn.Linear(input_dim, 1)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples=None):
        return True, nn.Sigmoid()(self.lin1(x).mul_(self.softmax_boost))
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x)
        batch_size = len(samples)
        log_probabilities = torch.log(proposal_output + util.epsilon)
        log_one_minus_probabilities = torch.log(1 - proposal_output + util.epsilon)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            if value > 0:
                l += log_probabilities[b]
            else:
                l += log_one_minus_probabilities[b]
        return l

class ProposalDiscrete(nn.Module):
    def __init__(self, input_dim, output_dim, softmax_boost=1.0):
        super(ProposalDiscrete, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples=None):
        return True, F.softmax(self.lin1(x).mul_(self.softmax_boost))
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x)
        batch_size = len(samples)
        log_weights = torch.log(proposal_output + util.epsilon)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            l += log_weights[b, int(value)]
        return l

class ProposalCategorical(nn.Module):
    def __init__(self, input_dim, output_dim, softmax_boost=1.0):
        super(ProposalCategorical, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.softmax_boost = softmax_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples=None):
        return True, F.softmax(self.lin1(x).mul_(self.softmax_boost))
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x)
        batch_size = len(samples)
        log_weights = torch.log(proposal_output + util.epsilon)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            l += log_weights[b, int(value)]
        return l

class ProposalUniformContinuous(nn.Module):
    def __init__(self, input_dim, softplus_boost=1.0):
        super(ProposalUniformContinuous, self).__init__()
        self.lin1 = nn.Linear(input_dim, 2)
        self.softplus_boost = softplus_boost
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
    def forward(self, x, samples):
        x = self.lin1(x)
        modes = x[:,0].unsqueeze(1)
        certainties = x[:,1].unsqueeze(1)
        modes = nn.Sigmoid()(modes)
        certainties = nn.Softplus()(certainties) * self.softplus_boost
        # check if mins are < maxs, if not, raise warning and return success = false
        prior_mins = Variable(util.Tensor([s.distribution.prior_min for s in samples]), requires_grad=False)
        prior_maxs = Variable(util.Tensor([s.distribution.prior_max for s in samples]), requires_grad=False)
        return True, torch.cat([(modes * (prior_maxs - prior_mins) + prior_mins), certainties], 1)
    def logpdf(self, x, samples):
        _, proposal_output = self.forward(x, samples)
        prior_mins = Variable(util.Tensor([s.distribution.prior_min for s in samples]), requires_grad=False)
        prior_maxs = Variable(util.Tensor([s.distribution.prior_max for s in samples]), requires_grad=False)
        batch_size = len(samples)
        modes = (proposal_output[:, 0] - prior_mins) / (prior_maxs - prior_mins)
        certainties = proposal_output[:, 1] + 2
        alphas = modes * (certainties - 2) + 1
        betas = (1 - modes) * (certainties - 2) + 1
        beta_funs = util.beta(alphas, betas)
        l = 0
        for b in range(batch_size):
            value = samples[b].value[0]
            prior_min = samples[b].distribution.prior_min
            prior_max = samples[b].distribution.prior_max
            normalized_value = (value - prior_min) / (prior_max - prior_min)
            alpha = alphas[b]
            beta = betas[b]
            beta_fun = beta_funs[b]
            l += (alpha - 1) * np.log(normalized_value + util.epsilon) + (beta - 1) * np.log(1 - normalized_value + util.epsilon) - torch.log(beta_fun + util.epsilon) - np.log(prior_max - prior_min + util.epsilon)
        return l

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

class ObserveEmbeddingCNN2D6C(nn.Module):
    def __init__(self, input_example_non_batch, output_dim, reshape=None):
        super(ObserveEmbeddingCNN2D6C, self).__init__()
        self.reshape = reshape
        if not self.reshape is None:
            input_example_non_batch = input_example_non_batch.view(self.reshape)
            self.reshape.insert(0, -1) # For correct handling of the batch dimension in self.forward
        if input_example_non_batch.dim() == 2:
            self.input_sample = input_example_non_batch.unsqueeze(0).cpu()
        elif input_example_non_batch.dim() == 3:
            self.input_sample = input_example_non_batch.cpu()
        else:
            util.log_error('ObserveEmbeddingCNN2D6C: Expecting a 3d input_example_non_batch (num_channels x height x width) or a 2d input_example_non_batch (height x width). Received: {0}'.format(input_example_non_batch.size()))
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
        if not self.reshape is None:
            x = x.view(self.reshape)
        if x.dim() == 3: # This indicates that there are no channel dimensions and we have BxHxW
            x = x.unsqueeze(1) # Add a channel dimension of 1 after the batch dimension so that we have BxCxHxW
        x = self.forward_cnn(x)
        x = x.view(-1, self.cnn_output_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x

class ObserveEmbeddingCNN3D4C(nn.Module):
    def __init__(self, input_example_non_batch, output_dim, reshape=None):
        super(ObserveEmbeddingCNN3D4C, self).__init__()
        self.reshape = reshape
        if not self.reshape is None:
            input_example_non_batch = input_example_non_batch.view(self.reshape)
            self.reshape.insert(0, -1) # For correct handling of the batch dimension in self.forward
        if input_example_non_batch.dim() == 3:
            self.input_sample = input_example_non_batch.unsqueeze(0).cpu()
        elif input_example_non_batch.dim() == 4:
            self.input_sample = input_example_non_batch.cpu()
        else:
            util.log_error('ObserveEmbeddingCNN3D4C: Expecting a 4d input_example_non_batch (num_channels x depth x height x width) or a 3d input_example_non_batch (depth x height x width). Received: {0}'.format(input_example_non_batch.size()))
        self.input_channels = self.input_sample.size(0)
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(self.input_channels, 64, 3)
        self.conv2 = nn.Conv3d(64, 64, 3)
        self.conv3 = nn.Conv3d(64, 128, 3)
        self.conv4 = nn.Conv3d(128, 128, 3)
    def configure(self):
        self.cnn_output_dim = self.forward_cnn(self.input_sample.unsqueeze(0)).view(-1).size(0)
        self.lin1 = nn.Linear(self.cnn_output_dim, self.output_dim)
        self.lin2 = nn.Linear(self.output_dim, self.output_dim)
    def forward_cnn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.MaxPool3d(2)(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = nn.MaxPool3d(2)(x)
        return x
    def forward(self, x):
        if not self.reshape is None:
            x = x.view(self.reshape)
        if x.dim() == 4: # This indicates that there are no channel dimensions and we have BxDxHxW
            x = x.unsqueeze(1) # Add a channel dimension of 1 after the batch dimension so that we have BxCxDxHxW
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
        self.one_hot_distribution = {}
        self.one_hot_address_dim = None
        self.one_hot_distribution_dim = None
        self.one_hot_address_empty = None
        self.one_hot_distribution_empty = None
        self.address_distributions = {}
        self.address_histogram = {}
        self.trace_length_histogram = {}
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
        self.trace_length_min = sys.maxsize
        self.trace_length_max = 0
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
        addresses = '; '.join(list(self.one_hot_address.keys()))
        distributions = ' '.join(list(self.one_hot_distribution.keys()))
        num_addresses = len(self.one_hot_address.keys())
        num_distributions = len(self.one_hot_distribution.keys())
        sum = 0
        total_count = 0
        for trace_length in self.trace_length_histogram:
            count = self.trace_length_histogram[trace_length]
            sum += trace_length * count
            total_count += count
        trace_length_mean = sum / total_count
        address_collisions = max(0, num_addresses - self.one_hot_address_dim)
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
                          colored('Num. addresses        : {0}'.format(num_addresses), 'yellow'),
                          colored('Address collisions    : {0}'.format(address_collisions), 'yellow'),
                          colored('Distributions         : {0}'.format(distributions), 'yellow'),
                          colored('Num. distributions    : {0}'.format(num_distributions), 'yellow'),
                          colored('Trace lengths seen    : min: {0}, max: {1}, mean: {2}'.format(self.trace_length_min, self.trace_length_max, trace_length_mean), 'yellow')])
        return info

    def polymorph(self, batch=None):
        if batch is None:
            batch = self.valid_batch

        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            example_trace_length = example_trace.length

            if example_trace_length > self.trace_length_max:
                self.trace_length_max = example_trace_length
            if example_trace_length < self.trace_length_min:
                self.trace_length_min = example_trace_length

            if example_trace_length in self.trace_length_histogram:
                self.trace_length_histogram[example_trace_length] += len(sub_batch)
            else:
                self.trace_length_histogram[example_trace_length] = len(sub_batch)

            for sample in example_trace.samples:
                address = sample.address
                # instance = sample.instance
                distribution = sample.distribution

                if address in self.address_histogram:
                    self.address_histogram[address] += 1
                else:
                    self.address_histogram[address] = 1

                if address not in self.address_distributions:
                    self.address_distributions[address] = distribution.name()

                # update the artifact's one-hot dictionary as needed
                self.add_one_hot_address(address)
                self.add_one_hot_distribution(distribution)

                # update the artifact's sample and proposal layers as needed
                if not address in self.sample_layers:
                    if self.smp_emb == 'fc':
                        sample_layer = SampleEmbeddingFC(sample.value.nelement(), self.smp_emb_dim)
                    else:
                        util.log_error('polymorph: Unsupported sample embedding: ' + self.smp_emb)
                    if isinstance(distribution, UniformDiscrete):
                        proposal_layer = ProposalUniformDiscrete(self.lstm_dim, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, Normal):
                        proposal_layer = ProposalNormal(self.lstm_dim)
                    elif isinstance(distribution, Flip):
                        proposal_layer = ProposalFlip(self.lstm_dim, self.softmax_boost)
                    elif isinstance(distribution, Discrete):
                        proposal_layer = ProposalDiscrete(self.lstm_dim, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, Categorical):
                        proposal_layer = ProposalCategorical(self.lstm_dim, distribution.prior_size, self.softmax_boost)
                    elif isinstance(distribution, UniformContinuous):
                        proposal_layer = ProposalUniformContinuous(self.lstm_dim, self.softmax_boost)
                    else:
                        util.log_error('polymorph: Unsupported distribution: ' + sample.distribution.name())
                    self.sample_layers[address] = sample_layer
                    self.proposal_layers[address] = proposal_layer
                    self.add_module('sample_layer({0})'.format(address), sample_layer)
                    self.add_module('proposal_layer({0})'.format(address), proposal_layer)
                    util.log_print(colored('Polymorphing, new layers attached : {0}'.format(util.truncate_str(address)), 'magenta', attrs=['bold']))
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

    def set_observe_embedding(self, example_observes, obs_emb, obs_emb_dim, obs_reshape=None):
        self.obs_emb = obs_emb
        self.obs_emb_dim = obs_emb_dim
        if obs_emb == 'fc':
            observe_layer = ObserveEmbeddingFC(Variable(example_observes), obs_emb_dim)
        elif obs_emb == 'cnn2d6c':
            observe_layer = ObserveEmbeddingCNN2D6C(Variable(example_observes), obs_emb_dim, obs_reshape)
            observe_layer.configure()
        elif obs_emb == 'cnn3d4c':
            observe_layer = ObserveEmbeddingCNN3D4C(Variable(example_observes), obs_emb_dim, obs_reshape)
            observe_layer.configure()
        elif obs_emb == 'lstm':
            observe_layer = ObserveEmbeddingLSTM(Variable(example_observes), obs_emb_dim)
        else:
            util.log_error('set_observe_embedding: Unsupported observation embedding: ' + obs_emb)

        self.observe_layer = observe_layer

    def set_lstm(self, lstm_dim, lstm_depth, dropout):
        self.lstm_dim = lstm_dim
        self.lstm_depth = lstm_depth
        self.lstm_input_dim = self.obs_emb_dim + self.smp_emb_dim + 2 * (self.one_hot_address_dim + self.one_hot_distribution_dim)
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_dim, lstm_depth, dropout=dropout)

    def set_one_hot_dims(self, one_hot_address_dim, one_hot_distribution_dim):
        self.one_hot_address_dim =one_hot_address_dim
        self.one_hot_distribution_dim = one_hot_distribution_dim
        self.one_hot_address_empty = Variable(util.Tensor(self.one_hot_address_dim).zero_(), requires_grad=False)
        self.one_hot_distribution_empty = Variable(util.Tensor(self.one_hot_distribution_dim).zero_(), requires_grad=False)

    def add_one_hot_address(self, address):
        if not address in self.one_hot_address:
            util.log_print(colored('Polymorphing, new address         : ' + util.truncate_str(address), 'magenta', attrs=['bold']))
            i = len(self.one_hot_address)
            if i < self.one_hot_address_dim:
                t = util.Tensor(self.one_hot_address_dim).zero_()
                t.narrow(0, i, 1).fill_(1)
                self.one_hot_address[address] = Variable(t, requires_grad=False)
            else:
                util.log_warning('Overflow (collision) in one_hot_address. Allowed: {0}; Encountered: {1}'.format(self.one_hot_address_dim, i + 1))
                self.one_hot_address[address] = random.choice(list(self.one_hot_address.values()))

    def add_one_hot_distribution(self, distribution):
        distribution_name = distribution.name()
        if not distribution_name in self.one_hot_distribution:
            util.log_print(colored('Polymorphing, new distribution    : ' + distribution_name, 'magenta', attrs=['bold']))
            i = len(self.one_hot_distribution)
            if i < self.one_hot_distribution_dim:
                t = util.Tensor(self.one_hot_distribution_dim).zero_()
                t.narrow(0, i, 1).fill_(1)
                self.one_hot_distribution[distribution_name] = Variable(t, requires_grad=False)
            else:
                util.log_warning('Overflow (collision) in one_hot_distribution. Allowed: {0}; Encountered: {1}'.format(self.one_hot_distribution_dim, i + 1))
                self.one_hot_distribution[distribution_name] = random.choice(list(self.one_hot_distribution.values()))

    def move_to_cuda(self, device_id=None):
        self.on_cuda = True
        self.cuda_device_id = device_id
        self.cuda(device_id)
        self.one_hot_address_empty = self.one_hot_address_empty.cuda(device_id)
        self.one_hot_distribution_empty = self.one_hot_distribution_empty.cuda(device_id)
        for k, t in self.one_hot_address.items():
            self.one_hot_address[k] = t.cuda(device_id)
        for k, t in self.one_hot_distribution.items():
            self.one_hot_distribution[k] = t.cuda(device_id)
        self.valid_batch.cuda(device_id)

    def move_to_cpu(self):
        self.on_cuda = False
        self.cpu()
        self.one_hot_address_empty = self.one_hot_address_empty.cpu()
        self.one_hot_distribution_empty = self.one_hot_distribution_empty.cpu()
        for k, t in self.one_hot_address.items():
            self.one_hot_address[k] = t.cpu()
        for k, t in self.one_hot_distribution.items():
            self.one_hot_distribution[k] = t.cpu()
        self.valid_batch.cpu()

    def valid_loss(self, data_parallel=False):
        return self.loss(self.valid_batch, data_parallel=data_parallel, volatile=True).data[0]

    def loss(self, batch, optimizer=None, truncate=-1, grad_clip=-1, data_parallel=False, volatile=False):
        gc.collect()

        example_observes = batch[0].observes
        if isinstance(self.observe_layer, ObserveEmbeddingLSTM):
            if example_observes.dim() != 2:
                util.log_error('loss: RNN observation embedding requires an observation shape of (T x F), where T is sequence length and F is feature length. Received observation with shape: {0}'.format(example_observes.size()))

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

        if truncate == -1:
            truncate = batch.traces_max_length
        loss = 0
        lstm_h = Variable(util.Tensor(self.lstm_depth, batch.length, self.lstm_dim).zero_(), requires_grad=False, volatile=volatile)
        lstm_c = lstm_h
        empty_lstm_input = Variable(util.Tensor(self.lstm_input_dim).zero_(), volatile=volatile)
        for time_step_start in range(0, batch.traces_max_length, truncate):
            time_step_end = min(time_step_start + truncate, batch.traces_max_length)
            # print('time_step_start', time_step_start)
            # print('time_step_end', time_step_end)
            trace_lengths_in_view = []
            for b in range(batch.length):
                trace = batch[b]
                if trace.length >= time_step_end:
                    trace_lengths_in_view.append(truncate)
                else:
                    if trace.length < time_step_start:
                        trace_lengths_in_view.append(0)
                    else:
                        trace_lengths_in_view.append(trace.length % truncate)
            trace_lengths_in_view = list(filter(lambda x: x != 0, trace_lengths_in_view))
            batch_length_in_view = len(trace_lengths_in_view)


            if isinstance(self.observe_layer, ObserveEmbeddingLSTM):
                obs_emb = self.observe_layer.forward_packed(obs_var, batch.length)
                obs_emb, _ = torch.nn.utils.rnn.pad_packed_sequence(obs_emb, batch_first=True)

                for b in range(batch.length):
                    seq_len = batch_sorted_by_observes[b].observes.size(0)
                    batch_sorted_by_observes[b].observes_embedding = obs_emb[b, seq_len - 1]
            else:
                obs = torch.cat([batch[b].observes for b in range(batch_length_in_view)]);
                if example_observes.dim() == 1:
                    obs = obs.view(batch_length_in_view, example_observes.size()[0])
                elif example_observes.dim() == 2:
                    obs = obs.view(batch_length_in_view, example_observes.size()[0], example_observes.size()[1])
                elif example_observes.dim() == 3:
                    obs = obs.view(batch_length_in_view, example_observes.size()[0], example_observes.size()[1], example_observes.size()[2])
                else:
                    util.log_error('loss: Unsupported observation dimensions: {0}'.format(example_observes.size()))
                obs_var = Variable(obs, requires_grad=False, volatile=volatile)

                if data_parallel and self.on_cuda:
                    obs_emb = torch.nn.DataParallel(self.observe_layer)(obs_var)
                else:
                    obs_emb = self.observe_layer(obs_var)

                for b in range(batch_length_in_view):
                    batch[b].observes_embedding = obs_emb[b]


            for sub_batch in batch.sub_batches:
                sub_batch_size = len(sub_batch)
                example_trace = sub_batch[0]
                for time_step in range(time_step_start, min(time_step_end, example_trace.length)):
                    current_sample = example_trace.samples[time_step]
                    current_address = current_sample.address
                    # current_instance = current_sample.instance
                    current_distribution = current_sample.distribution

                    current_one_hot_address = self.one_hot_address[current_address]
                    current_one_hot_distribution = self.one_hot_distribution[current_distribution.name()]

                    if time_step == 0:
                        prev_sample_embedding = Variable(util.Tensor(sub_batch_size, self.smp_emb_dim).zero_(), requires_grad=False, volatile=volatile)

                        prev_one_hot_address = self.one_hot_address_empty
                        prev_one_hot_distribution = self.one_hot_distribution_empty
                    else:
                        prev_sample = example_trace.samples[time_step - 1]
                        prev_address = prev_sample.address
                        prev_distribution = prev_sample.distribution
                        smp = torch.cat([sub_batch[b].samples[time_step - 1].value for b in range(sub_batch_size)]).view(sub_batch_size, prev_sample.value.nelement())

                        smp_var = Variable(smp, requires_grad=False, volatile=volatile)
                        sample_layer = self.sample_layers[prev_address]
                        # if data_parallel and self.on_cuda:
                        #     prev_sample_embedding = torch.nn.DataParallel(sample_layer)(smp_var)
                        # else:
                        prev_sample_embedding = sample_layer(smp_var)

                        prev_one_hot_address = self.one_hot_address[prev_address]
                        prev_one_hot_distribution = self.one_hot_distribution[prev_distribution.name()]

                    for b in range(sub_batch_size):
                        t = torch.cat([sub_batch[b].observes_embedding,
                                       prev_sample_embedding[b],
                                       prev_one_hot_address,
                                       prev_one_hot_distribution,
                                       current_one_hot_address,
                                       current_one_hot_distribution])
                        sub_batch[b].samples[time_step].lstm_input = t


            lstm_input = []
            for time_step in range(time_step_start, time_step_end):
                t = []
                for b in range(batch_length_in_view):
                    trace = batch[b]
                    if time_step < trace.length:
                        t.append(trace.samples[time_step].lstm_input)
                    else:
                        t.append(empty_lstm_input)
                t = torch.cat(t).view(batch_length_in_view, -1)
                lstm_input.append(t)
            lstm_input = torch.cat(lstm_input).view(time_step_end - time_step_start, batch_length_in_view, -1)


            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, trace_lengths_in_view)
            lstm_h = lstm_h.detach()[:, 0:batch_length_in_view].contiguous()
            lstm_c = lstm_c.detach()[:, 0:batch_length_in_view].contiguous()
            if data_parallel and self.on_cuda:
                lstm_output, (lstm_h, lstm_c) = torch.nn.DataParallel(self.lstm, dim=1)(lstm_input, (lstm_h, lstm_c))
            else:
                lstm_output, (lstm_h, lstm_c) = self.lstm(lstm_input, (lstm_h, lstm_c))
            lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output)

            for b in range(batch.length):
                trace = batch[b]
                for time_step in range(time_step_start, min(time_step_end, trace.length)):
                    trace.samples[time_step].lstm_output = lstm_output[time_step - time_step_start, b]


            logpdf = 0
            for sub_batch in batch.sub_batches:
                sub_batch_size = len(sub_batch)
                example_trace = sub_batch[0]

                for time_step in range(time_step_start, min(time_step_end, example_trace.length)):
                    current_sample = example_trace.samples[time_step]
                    current_address = current_sample.address
                    # current_instance = current_sample.instance

                    p = []
                    for b in range(sub_batch_size):
                        p.append(sub_batch[b].samples[time_step].lstm_output)
                    p = torch.cat(p).view(sub_batch_size, -1)
                    proposal_input = p

                    current_samples = [sub_batch[b].samples[time_step] for b in range(sub_batch_size)]
                    proposal_layer = self.proposal_layers[current_address]
                    logpdf += proposal_layer.logpdf(proposal_input, current_samples)

            logpdf = -logpdf / batch.length

            if not optimizer is None:
                optimizer.zero_grad()
                logpdf.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm(self.parameters(), grad_clip)
                optimizer.step()

            loss += logpdf

        return loss
