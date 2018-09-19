import torch
import torch.nn as nn
import torch.optim as optim
import sys
import gc
import time
import os
import shutil
import uuid
import tempfile
import tarfile
import copy
from threading import Thread
from termcolor import colored

from . import EmbeddingFeedForward, ProposalNormalNormalMixture, ProposalUniformTruncatedNormalMixture, ProposalCategoricalCategorical, ProposalPoissonTruncatedNormalMixture
from .. import __version__, util, ObserveEmbedding
from ..distributions import Normal, Uniform, Categorical, Poisson


class InferenceNetworkFeedForward(nn.Module):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FULLY_CONNECTED, 'dim': 32, 'depth': 2}}
    def __init__(self, model, valid_batch_size=64, observe_embeddings={}):
        super().__init__()
        self._model = model
        self._layer_proposal = nn.ModuleDict()
        self._layer_observe_embedding = nn.ModuleDict()
        self._layer_observe_embedding_final = None
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
        self._modified = util.get_time_str()
        self._updates = 0
        self._on_cuda = False
        self._device = torch.device('cpu')

        self._valid_batch_size = valid_batch_size
        self._observe_embeddings = observe_embeddings
        self._valid_batch = None

    def _init_layer_observe_embeddings(self, observe_embeddings):
        if len(observe_embeddings) == 0:
            raise ValueError('At least one observe embedding is needed to initialize inference network.')
        observe_embedding_total_dim = 0
        for name, value in observe_embeddings.items():
            distribution = self._valid_batch.traces[0].named_variables[name].distribution
            if distribution is None:
                raise ValueError('Observable {}: cannot use this observation as an input to the inference network, because there is no associated likelihood.'.format(name))
            else:
                input_shape = distribution.sample().size()
            if 'dim' in value:
                output_shape = torch.Size([value['dim']])
            else:
                print('Observable {}: embedding dim not specified, using the default 256.'.format(name))
                output_shape = torch.Size([256])
            if 'depth' in value:
                depth = value['depth']
            else:
                print('Observable {}: embedding depth not specified, using the default 2.'.format(name))
                depth = 2
            if 'embedding' in value:
                embedding = value['embedding']
            else:
                print('Observable {}: observe embedding not specified, using the default FULLY_CONNECTED.'.format(name))
                embedding = ObserveEmbedding.FULLY_CONNECTED
            if embedding == ObserveEmbedding.FULLY_CONNECTED:
                layer = EmbeddingFeedForward(input_shape=input_shape, output_shape=output_shape, num_layers=depth)
            else:
                raise ValueError('Unknown embedding: {}'.format(embedding))
            self._layer_observe_embedding[name] = layer
            observe_embedding_total_dim += util.prod(output_shape)
        self._layer_hidden_shape = torch.Size([observe_embedding_total_dim])
        self._layer_observe_embedding_final = EmbeddingFeedForward(input_shape=self._layer_hidden_shape, output_shape=self._layer_hidden_shape, num_layers=1)

    def _save(self, file_name):
        self._modified = util.get_time_str()
        self._updates += 1

        data = {}
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        data['inference_network'] = copy.copy(self)
        data['inference_network']._model = None
        data['inference_network']._optimizer = None

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_inference_network')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_inference_network')
            tar.close()
            shutil.rmtree(tmp_dir)
        t = Thread(target=thread_save)
        t.start()
        t.join()

    @staticmethod
    def _load(file_name):
        try:
            tar = tarfile.open(file_name, 'r:gz')
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file = os.path.join(tmp_dir, 'pyprob_inference_network')
            tar.extract('pyprob_inference_network', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
                shutil.rmtree(tmp_dir)
        except:
            raise RuntimeError('Cannot load inference network.')

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded network: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded network: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        ret = data['inference_network']
        if util._cuda_enabled:
            if ret._on_cuda:
                if ret._device != util._device:
                    print(colored('Warning: loading CUDA (device {}) network to CUDA (device {})'.format(ret._device, util._device), 'red', attrs=['bold']))
            else:
                print(colored('Warning: loading CPU network to CUDA (device {})'.format(util._device), 'red', attrs=['bold']))
        else:
            if ret._on_cuda:
                print(colored('Warning: loading CUDA (device {}) network to CPU'.format(ret._device), 'red', attrs=['bold']))
        ret.to(device=util._device)
        return ret

    def to(self, device=None, *args, **kwargs):
        self._device = device
        if 'cuda' in str(device):
            self._on_cuda = True
        super().to(device=device, *args, *kwargs)

    def _embed_observe(self, traces=None):
        embedding = []
        for name, layer in self._layer_observe_embedding.items():
            values = torch.stack([trace.named_variables[name].value for trace in traces]).view(len(traces), -1)
            embedding.append(layer(values))
        embedding = torch.cat(embedding, dim=1)
        embedding = self._layer_observe_embedding_final(embedding)
        return embedding

    def infer_trace_init(self, observe=None):
        self._infer_observe = observe
        embedding = []
        for name, layer in self._layer_observe_embedding.items():
            value = util.to_tensor(observe[name]).view(1, -1)
            embedding.append(layer(value))
        embedding = torch.cat(embedding, dim=1)
        self._infer_observe_embedding = self._layer_observe_embedding_final(embedding)

    def infer_trace_step(self, variable, previous_variable=None):
        success = True
        address = variable.address
        distribution = variable.distribution
        if address not in self._layer_proposal:
            print('Warning: no proposal layer for: {}'.format(address))
            success = False

        if success:
            proposal_distribution = self._layer_proposal[address].forward(self._infer_observe_embedding, [variable])
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
                        layer = ProposalNormalNormalMixture(self._layer_hidden_shape, variable_shape)
                    elif isinstance(distribution, Uniform):
                        layer = ProposalUniformTruncatedNormalMixture(self._layer_hidden_shape, variable_shape)
                    elif isinstance(distribution, Poisson):
                        layer = ProposalPoissonTruncatedNormalMixture(self._layer_hidden_shape, variable_shape)
                    elif isinstance(distribution, Categorical):
                        layer = ProposalCategoricalCategorical(self._layer_hidden_shape, distribution.num_categories)
                    else:
                        raise RuntimeError('Distribution currently unsupported: {}'.format(distribution.name))
                    self._layer_proposal[address] = layer
                    layers_changed = True
        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total number of parameters: {:,}'.format(num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def _loss(self, batch):
        gc.collect()
        batch_loss = 0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            observe_embedding = self._embed_observe(sub_batch)
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

    def optimize(self, num_traces, batch_generator, batch_size=64, valid_interval=1000, learning_rate=0.0001, weight_decay=1e-5, auto_save_file_name=None, auto_save_interval_sec=600, *args, **kwargs):
        if self._valid_batch is None:
            print('Initializing inference network...')
            self._valid_batch = batch_generator.get_batch(self._valid_batch_size, discard_source=True)
            self._init_layer_observe_embeddings(self._observe_embeddings)
            self._polymorph(self._valid_batch)

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
        loss_min_str = ''
        time_since_loss_min_str = ''
        last_auto_save_time = time.time() - auto_save_interval_sec
        while not stop:
            iteration += 1
            batch = batch_generator.get_batch(batch_size)
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

                if auto_save_file_name is not None:
                    if time.time() - last_auto_save_time > auto_save_interval_sec:
                        last_auto_save_time = time.time()
                        file_name = auto_save_file_name + '_' + util.get_time_stamp()
                        print('\rSaving to disk...', end='\r')
                        self._save(file_name)

                print_line = '{} | {} | {} | {} | {} | {} | {}'.format(total_training_seconds_str, total_training_traces_str, loss_initial_str, loss_min_str, loss_str, time_since_loss_min_str, traces_per_second_str)
                max_print_line_len = max(len(print_line), max_print_line_len)
                print(print_line.ljust(max_print_line_len), end='\r')
                sys.stdout.flush()
        print()
