import torch
import torch.nn as nn
import torch.optim as optim
import sys
import gc
import time
from termcolor import colored

from . import EmbeddingFeedForward, ProposalNormal, ProposalUniform
from .. import util, TraceMode, ObserveEmbedding
from ..distributions import Normal, Uniform


class Batch():
    def __init__(self, traces):
        self.batch = traces
        self.length = len(traces)
        sub_batches = {}
        for trace in traces:
            if trace.length == 0:
                raise ValueError('Trace of length zero.')
            trace_hash = ''.join([variable.address for variable in trace.variables_controlled])
            if trace_hash not in sub_batches:
                sub_batches[trace_hash] = []
            sub_batches[trace_hash].append(trace)
        self.sub_batches = list(sub_batches.values())


class InferenceNetworkFeedForward(nn.Module):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FULLY_CONNECTED, 'dim': 32, 'depth': 2}}
    def __init__(self, model, prior_inflation, valid_batch_size=64, observe_embeddings={}):
        super().__init__()
        self._model = model
        self._prior_inflation = prior_inflation
        self._layer_observe_embedding = {}
        self._layer_proposal = {}
        self._layer_hidden_shape = None
        self._infer_observe = None
        self._infer_observe_embedding = {}
        self._optimizer = None

        self._total_train_seconds = 0
        self._total_train_traces = 0
        self._total_train_iterations = 0
        self._loss_initial = None
        self._loss_min = float('inf')
        self._loss_max = None
        self._loss_previous = float('inf')
        self._history_train_loss = []
        self._history_train_loss_trace = []
        self._history_valid_loss = []
        self._history_valid_loss_trace = []
        self._history_num_params = []
        self._history_num_params_trace = []

        self._valid_batch = self._get_batch(valid_batch_size)
        self._init_layer_observe_embeddings(observe_embeddings)
        self._polymorph(self._valid_batch)

    def _init_layer_observe_embeddings(self, observe_embeddings):
        if len(observe_embeddings) == 0:
            raise ValueError('At least one observe embedding is needed to initialize inference network.')
        self._layer_observe_embeddings = {}
        observe_embedding_total_dim = 0
        for name, value in observe_embeddings.items():
            distribution = self._valid_batch.batch[0].named_variables[name].distribution
            if distribution is None:
                raise ValueError('Observable {}: cannot use this observation as an input to the inference network, because there is no associated likelihood.'.format(name))
            else:
                input_shape = distribution.sample().size()
            if 'dim' in value:
                output_shape = torch.Size([value['dim']])
            else:
                print('Observable {}: embedding dim not provided, using the default 256.'.format(name))
                output_shape = torch.Size([256])
            if 'depth' in value:
                depth = value['depth']
            else:
                print('Observable {}: embedding depth not provided, using the default 2.'.format(name))
                depth = 2
            if 'embedding' in value:
                embedding = value['embedding']
            else:
                print('Observable {}: observe embedding not provided, using the default FULLY_CONNECTED.'.format(name))
                embedding = ObserveEmbedding.FULLY_CONNECTED
            if embedding == ObserveEmbedding.FULLY_CONNECTED:
                layer = EmbeddingFeedForward(input_shape=input_shape, output_shape=output_shape, num_layers=depth)
            else:
                raise ValueError('Unknown embedding: {}'.format(embedding))
            self._layer_observe_embedding[name] = layer
            self.add_module('_layer_observe_embedding({})'.format(name), layer)
            observe_embedding_total_dim += util.prod(output_shape)
        self._layer_hidden_shape = torch.Size([observe_embedding_total_dim])
        self._layer_observe_embedding_final = EmbeddingFeedForward(input_shape=self._layer_hidden_shape, output_shape=self._layer_hidden_shape, num_layers=1)

    def _get_batch(self, length=64, *args, **kwargs):
        traces, _ = self._model._traces(length, trace_mode=TraceMode.PRIOR, prior_inflation=self._prior_inflation, silent=True, *args, **kwargs)
        return Batch(traces)

    def _embed_observe(self, observe=None):
        if observe is None:
            raise ValueError('All observes in observe_embeddings are needed to initialize a new trace.')
        return self._layer_observe_embedding_final(torch.cat([layer.forward(observe[name]).view(-1) for name, layer in self._layer_observe_embedding.items()]))

    def infer_trace_init(self, observe=None):
        self._infer_observe = observe
        self._infer_observe_embedding = self._embed_observe({name: util.to_tensor(v) for name, v in observe.items()})

    def infer_trace_step(self, variable, previous_variable=None):
        success = True
        address = variable.address
        distribution = variable.distribution
        if address not in self._layer_proposal:
            print('Warning: no proposal layer for: {}'.format(address))
            success = False

        if success:
            proposal_distribution = self._layer_proposal[address].forward(self._infer_observe_embedding.unsqueeze(0), [variable])
            return proposal_distribution
        else:
            print('Warning: no proposal can be made, prior will be used.')
            return distribution

    def _polymorph(self, batch):
        layers_changed = False
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            for variable in example_trace.variables_controlled:
                address = variable.address
                distribution = variable.distribution
                variable_shape = variable.value.shape
                if address not in self._layer_proposal:
                    print('New proposal layer for address: {}'.format(util.truncate_str(address)))
                    if isinstance(distribution, Normal):
                        layer = ProposalNormal(self._layer_hidden_shape, variable_shape)
                    elif isinstance(distribution, Uniform):
                        layer = ProposalUniform(self._layer_hidden_shape, variable_shape)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    self._layer_proposal[address] = layer
                    self.add_module('layer_proposal({})'.format(address), layer)
                    layers_changed = True
        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('New number of parameters: {:,}'.format(num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def _loss(self, batch):
        gc.collect()
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            observe_embedding = torch.stack([self._embed_observe({name: variable.value for name, variable in trace.named_variables.items()}) for trace in sub_batch])
            sub_batch_loss = 0.
            for time_step in range(example_trace.length_controlled):
                address = example_trace.variables_controlled[time_step].address
                variables = [trace.variables_controlled[time_step] for trace in sub_batch]
                values = torch.stack([v.value for v in variables])
                proposal_distribution = self._layer_proposal[address].forward(observe_embedding, variables)
                log_prob = proposal_distribution.log_prob(values)
                if util.has_nan_or_inf(log_prob):
                    print(colored('Warning: NaN, -Inf, or Inf encountered in proposal log_prob.', 'red', attrs=['bold']))
                    print('proposal_distribution', proposal_distribution)
                    print('values', values)
                    print('log_prob', log_prob)
                    print('Fixing -Inf')
                    log_prob = util.replace_negative_inf(log_prob)
                    print('log_prob', log_prob)
                    if util.has_nan_or_inf(log_prob):
                        print(colored('Nan or Inf present in proposal log_prob.', 'red', attrs=['bold']))
                        return False, 0
                sub_batch_loss += -torch.sum(log_prob)
            batch_loss += sub_batch_loss
        return True, batch_loss / batch.length

    def optimize(self, num_traces=None, batch_size=64, valid_interval=1000, learning_rate=0.0001, weight_decay=1e-5, *args, **kwargs):
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time.time()
        time_last_batch = time.time()
        last_validation_trace = -valid_interval + 1
        iteration = 0
        trace = 0
        stop = False
        print('Train. time | Trace     | Init. loss| Min. loss | Curr. loss| T.since min | Traces/sec')
        max_print_line_len = 0
        while not stop:
            iteration += 1
            batch = self._get_batch(batch_size)
            layers_changed = self._polymorph(batch)

            if (self._optimizer is None) or layers_changed:
                self._optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

            self._optimizer.zero_grad()
            success, loss = self._loss(batch)
            if not success:
                print(colored('Cannot compute loss, skipping batch. Loss: {}'.format(loss), 'red', attrs=['bold']))
            else:
                loss.backward()
                self._optimizer.step()
                loss = float(loss)

                if self._loss_initial is None:
                    self._loss_initial = loss
                    self._loss_max = loss
                loss_initial_str = '{:+.2e}'.format(self._loss_initial)
                # loss_max_str = '{:+.3e}'.format(self._loss_max)
                if loss < self._loss_min:
                    self._loss_min = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'green', attrs=['bold'])
                    loss_min_str = colored('{:+.2e}'.format(self._loss_min), 'green', attrs=['bold'])
                    time_loss_min = time.time()
                    time_since_loss_min_str = colored(util.days_hours_mins_secs_str(0), 'green', attrs=['bold'])
                elif loss > self._loss_max:
                    self._loss_max = loss
                    loss_str = colored('{:+.2e}'.format(loss), 'red', attrs=['bold'])
                    # loss_max_str = colored('{:+.3e}'.format(self._loss_max), 'red', attrs=['bold'])
                else:
                    if loss < self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'green')
                    elif loss > self._loss_previous:
                        loss_str = colored('{:+.2e}'.format(loss), 'red')
                    else:
                        loss_str = '{:+.2e}'.format(loss)
                    loss_min_str = '{:+.2e}'.format(self._loss_min)
                    # loss_max_str = '{:+.3e}'.format(self._loss_max)
                    time_since_loss_min_str = util.days_hours_mins_secs_str(time.time() - time_loss_min)

                self._loss_previous = loss
                self._total_train_iterations += 1
                trace += batch.length
                self._total_train_traces += batch.length
                total_training_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                self._total_train_seconds = prev_total_train_seconds + (time.time() - time_start)
                total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                traces_per_second_str = '{:,.1f}'.format(int(batch.length / (time.time() - time_last_batch)))
                time_last_batch = time.time()
                if num_traces is not None:
                    if trace >= num_traces:
                        stop = True

                self._history_train_loss.append(loss)
                self._history_train_loss_trace.append(self._total_train_traces)
                if trace - last_validation_trace > valid_interval:
                    print('\rComputing validation loss...', end='\r')
                    with torch.no_grad():
                        _, valid_loss = self._loss(self._valid_batch)
                    valid_loss = float(valid_loss)
                    self._history_valid_loss.append(valid_loss)
                    self._history_valid_loss_trace.append(self._total_train_traces)
                    last_validation_trace = trace - 1
                    # if auto_save:
                    #     file_name = auto_save_file_name + '_' + util.get_time_stamp()
                    #     print('\rSaving to disk...', end='\r')
                    #     self._save(file_name)

                print_line = '{} | {} | {} | {} | {} | {} | {}'.format(total_training_seconds_str, total_training_traces_str, loss_initial_str, loss_min_str, loss_str, time_since_loss_min_str, traces_per_second_str)
                max_print_line_len = max(len(print_line), max_print_line_len)
                print(print_line.ljust(max_print_line_len), end='\r')
                sys.stdout.flush()
