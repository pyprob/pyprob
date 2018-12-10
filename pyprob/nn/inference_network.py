import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import sys
import time
import os
import shutil
import uuid
import tempfile
import tarfile
import copy
from threading import Thread
from termcolor import colored

from . import Batch, OfflineDataset, EmbeddingFeedForward, EmbeddingCNN2D5C, EmbeddingCNN3D5C
from .. import __version__, util, Optimizer, ObserveEmbedding


class InferenceNetwork(nn.Module):
    # observe_embeddings example: {'obs1': {'embedding':ObserveEmbedding.FEEDFORWARD, 'reshape': [10, 10], 'dim': 32, 'depth': 2}}
    def __init__(self, model, observe_embeddings={}, network_type=''):
        super().__init__()
        self._model = model
        self._layers_observe_embedding = nn.ModuleDict()
        self._layers_observe_embedding_final = None
        self._layers_pre_generated = False
        self._layers_initialized = False
        self._observe_embeddings = observe_embeddings
        self._observe_embedding_dim = None
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
        self._distributed_train_loss = util.to_tensor(0.)
        self._distributed_valid_loss = util.to_tensor(0.)
        self._distributed_train_loss_min = float('inf')
        self._distributed_valid_loss_min = float('inf')
        self._distributed_history_train_loss = []
        self._distributed_history_train_loss_trace = []
        self._distributed_history_valid_loss = []
        self._distributed_history_valid_loss_trace = []
        self._modified = util.get_time_str()
        self._updates = 0
        self._on_cuda = False
        self._device = torch.device('cpu')
        self._network_type = network_type
        self._optimizer_type = None
        self._learning_rate = None
        self._momentum = None
        self._batch_size = None
        self._distributed_backend = None
        self._distributed_world_size = None


    def _init_layers_observe_embedding(self, observe_embeddings, example_trace):
        if len(observe_embeddings) == 0:
            raise ValueError('At least one observe embedding is needed to initialize inference network.')
        observe_embedding_total_dim = 0
        for name, value in observe_embeddings.items():
            variable = example_trace.named_variables[name]
            # distribution = variable.distribution
            # if distribution is None:
            #     raise ValueError('Observable {}: cannot use this observation as an input to the inference network, because there is no associated likelihood.'.format(name))
            # else:
            if 'reshape' in value:
                input_shape = torch.Size(value['reshape'])
            else:
                input_shape = variable.value.size()
            if 'dim' in value:
                output_shape = torch.Size([value['dim']])
            else:
                print('Observable {}: embedding dim not specified, using the default 256.'.format(name))
                output_shape = torch.Size([256])
            if 'embedding' in value:
                embedding = value['embedding']
            else:
                print('Observable {}: observe embedding not specified, using the default FEEDFORWARD.'.format(name))
                embedding = ObserveEmbedding.FEEDFORWARD
            if embedding == ObserveEmbedding.FEEDFORWARD:
                if 'depth' in value:
                    depth = value['depth']
                else:
                    print('Observable {}: embedding depth not specified, using the default 2.'.format(name))
                    depth = 2
                layer = EmbeddingFeedForward(input_shape=input_shape, output_shape=output_shape, num_layers=depth)
            elif embedding == ObserveEmbedding.CNN2D5C:
                layer = EmbeddingCNN2D5C(input_shape=input_shape, output_shape=output_shape)
            elif embedding == ObserveEmbedding.CNN3D5C:
                layer = EmbeddingCNN3D5C(input_shape=input_shape, output_shape=output_shape)
            else:
                raise ValueError('Unknown embedding: {}'.format(embedding))
            layer.to(device=util._device)
            self._layers_observe_embedding[name] = layer
            observe_embedding_total_dim += util.prod(output_shape)
        self._observe_embedding_dim = observe_embedding_total_dim
        print('Observe embedding dimension: {}'.format(self._observe_embedding_dim))
        self._layers_observe_embedding_final = EmbeddingFeedForward(input_shape=self._observe_embedding_dim, output_shape=self._observe_embedding_dim, num_layers=2)
        self._layers_observe_embedding_final.to(device=util._device)

    def _embed_observe(self, traces=None):
        embedding = []
        for name, layer in self._layers_observe_embedding.items():
            values = torch.stack([util.to_tensor(trace.named_variables[name].value) for trace in traces]).view(len(traces), -1)
            embedding.append(layer(values))
        embedding = torch.cat(embedding, dim=1)
        embedding = self._layers_observe_embedding_final(embedding)
        return embedding

    def _infer_init(self, observe=None):
        self._infer_observe = observe
        embedding = []
        for name, layer in self._layers_observe_embedding.items():
            value = util.to_tensor(observe[name]).view(1, -1)
            embedding.append(layer(value))
        embedding = torch.cat(embedding, dim=1)
        self._infer_observe_embedding = self._layers_observe_embedding_final(embedding)

    def _init_layers(self):
        raise NotImplementedError()

    def _polymorph(self, batch):
        raise NotImplementedError()

    def _infer_step(self, variable, previous_variable=None, proposal_min_train_iterations=None):
        raise NotImplementedError()

    def _loss(self, batch):
        raise NotImplementedError()

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
        self._on_cuda = 'cuda' in str(device)
        super().to(device=device, *args, *kwargs)

    def _pre_generate_layers(self, dataset, batch_size=64, save_file_name_prefix=None):
        if not self._layers_initialized:
            self._init_layers_observe_embedding(self._observe_embeddings, example_trace=dataset.__getitem__(0))
            self._init_layers()
            self._layers_initialized = True

        self._layers_pre_generated = True
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: Batch(x))
        util.progress_bar_init('Layer pre-generation...', len(dataset), 'Traces')
        i = 0
        for i_batch, batch in enumerate(dataloader):
            i += len(batch)
            layers_changed = self._polymorph(batch)
            util.progress_bar_update(i)
            if layers_changed and (save_file_name_prefix is not None):
                file_name = '{}_00000000_pre_generated.network'.format(save_file_name_prefix)
                print('\rSaving to disk...  ', end='\r')
                self._save(file_name)
        util.progress_bar_end('Layer pre-generation complete')

    def _distributed_sync_parameters(self):
        """ broadcast rank 0 parameter to all ranks """
        # print('Distributed training synchronizing parameters across nodes...')
        for param in self.parameters():
            dist.broadcast(param.data, 0)

    def _distributed_zero_grad(self):
        # Create zero tensors for gradients not initialized at this distributed training rank
        # print('Distributed zeroing gradients...')
        for param in self.parameters():
            # if (param.grad is None):
            param.grad = util.to_tensor(torch.zeros_like(param.data))

    def _distributed_sync_grad(self, world_size):
        """ all_reduce grads from all ranks """
        # print('Distributed training synchronizing gradients across nodes...')
        for param in self.parameters():
            try:
                dist.all_reduce(param.grad.data)
                param.grad.data /= float(world_size)  # average gradients
            except AttributeError:
                # pass
                print('None for grad, with param.size=', param.size())

    def _distributed_update_train_loss(self, loss, world_size):
        self._distributed_train_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_train_loss)
        self._distributed_train_loss /= float(world_size)
        self._distributed_history_train_loss.append(float(self._distributed_train_loss))
        self._distributed_history_train_loss_trace.append(self._total_train_traces)
        if float(self._distributed_train_loss) < self._distributed_train_loss_min:
            self._distributed_train_loss_min = float(self._distributed_train_loss)
        print(colored('Distributed mean train. loss across ranks : {:+.2e}, min. train. loss: {:+.2e}'.format(self._distributed_train_loss, self._distributed_train_loss_min), 'yellow', attrs=['bold']))

    def _distributed_update_valid_loss(self, loss, world_size):
        self._distributed_valid_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_valid_loss)
        self._distributed_valid_loss /= float(world_size)
        self._distributed_history_valid_loss.append(float(self._distributed_valid_loss))
        self._distributed_history_valid_loss_trace.append(self._total_train_traces)
        if float(self._distributed_valid_loss) < self._distributed_valid_loss_min:
            self._distributed_valid_loss_min = float(self._distributed_valid_loss)
        print(colored('Distributed mean valid. loss across ranks : {:+.2e}, min. valid. loss: {:+.2e}'.format(self._distributed_valid_loss, self._distributed_valid_loss_min), 'yellow', attrs=['bold']))

    def optimize(self, num_traces, dataset, dataset_valid, batch_size=64, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate=0.0001, momentum=0.9, weight_decay=1e-5, save_file_name_prefix=None, save_every_sec=600, distributed_backend=None, distributed_params_sync_every=10000, distributed_loss_update_every=None, dataloader_offline_num_workers=0, *args, **kwargs):
        if not self._layers_initialized:
            self._init_layers_observe_embedding(self._observe_embeddings, example_trace=dataset.__getitem__(0))
            self._init_layers()
            self._layers_initialized = True

        if distributed_backend is None:
            distributed_world_size = 1
            distributed_rank = 0
        else:
            dist.init_process_group(backend=distributed_backend)
            distributed_world_size = dist.get_world_size()
            distributed_rank = dist.get_rank()
            util.init_distributed_print(distributed_rank, distributed_world_size, False)
            print(colored('Distributed synchronous training', 'yellow', attrs=['bold']))
            print(colored('Distributed backend       : {}'.format(distributed_backend), 'yellow', attrs=['bold']))
            print(colored('Distributed world size    : {}'.format(distributed_world_size), 'yellow', attrs=['bold']))
            print(colored('Distributed minibatch size: {} (global), {} (per node)'.format(batch_size * distributed_world_size, batch_size), 'yellow', attrs=['bold']))
            print(colored('Distributed learning rate : {} (global), {} (base)'.format(learning_rate * distributed_world_size, learning_rate), 'yellow', attrs=['bold']))
            print(colored('Distributed optimizer     : {}'.format(str(optimizer_type)), 'yellow', attrs=['bold']))
            self._distributed_backend = distributed_backend
            self._distributed_world_size = distributed_world_size

        self._optimizer_type = optimizer_type
        self._batch_size = batch_size
        self._learning_rate = learning_rate * distributed_world_size
        self._momentum = momentum
        self.train()
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time.time()
        time_last_batch = time.time()
        if valid_every is None:
            valid_every = max(100, num_traces / 1000)
        if distributed_loss_update_every is None:
            distributed_loss_update_every = valid_every
        last_validation_trace = -valid_every + 1
        epoch = 0
        iteration = 0
        trace = 0
        stop = False
        print('Train. time | Epoch| Trace     | Init. loss| Min. loss | Curr. loss| T.since min | Traces/sec')
        max_print_line_len = 0
        loss_min_str = ''
        time_since_loss_min_str = ''
        last_auto_save_time = time.time() - save_every_sec
        num_workers = 0
        if isinstance(dataset, OfflineDataset):  # and (distributed_world_size == 1):
            num_workers = dataloader_offline_num_workers
        # print('num_workers', num_workers)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: Batch(x))
        if dataset_valid is not None:
            dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x: Batch(x))
        while not stop:
            epoch += 1
            for i_batch, batch in enumerate(dataloader):
                # Important, a self._distributed_sync_parameters() needs to happen at the very beginning of a training
                if (distributed_world_size > 1) and (iteration % distributed_params_sync_every == 0):
                    self._distributed_sync_parameters()

                if self._layers_pre_generated:  # and (distributed_world_size > 1):
                    layers_changed = False
                else:
                    layers_changed = self._polymorph(batch)

                if (self._optimizer is None) or layers_changed:
                    if optimizer_type == Optimizer.ADAM:
                        self._optimizer = optim.Adam(self.parameters(), lr=learning_rate * distributed_world_size, weight_decay=weight_decay)
                    else:  # optimizer_type == Optimizer.SGD
                        self._optimizer = optim.SGD(self.parameters(), lr=learning_rate * distributed_world_size, momentum=momentum, nesterov=True, weight_decay=weight_decay)
                # self._optimizer.zero_grad()
                if distributed_world_size > 1:
                    self._distributed_zero_grad()
                else:
                    self._optimizer.zero_grad()
                success, loss = self._loss(batch)
                if not success:
                    print(colored('Cannot compute loss, skipping batch. Loss: {}'.format(loss), 'red', attrs=['bold']))
                else:
                    loss.backward()
                    if distributed_world_size > 1:
                        self._distributed_sync_grad(distributed_world_size)
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
                    trace += batch.size
                    self._total_train_traces += batch.size * distributed_world_size
                    total_train_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                    epoch_str = '{:4}'.format('{:,}'.format(epoch))
                    self._total_train_seconds = prev_total_train_seconds + (time.time() - time_start)
                    total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                    traces_per_second_str = '{:,.1f}'.format(int(batch.size * distributed_world_size / (time.time() - time_last_batch)))
                    time_last_batch = time.time()
                    if num_traces is not None:
                        if trace >= num_traces:
                            stop = True

                    self._history_train_loss.append(loss)
                    self._history_train_loss_trace.append(self._total_train_traces)
                    if dataset_valid is not None:
                        if trace - last_validation_trace > valid_every:
                            print('\rComputing validation loss...  ', end='\r')
                            valid_loss = 0
                            with torch.no_grad():
                                for i_batch, batch in enumerate(dataloader_valid):
                                    _, v = self._loss(batch)
                                    valid_loss += v
                            valid_loss = float(valid_loss / len(dataset_valid))
                            self._history_valid_loss.append(valid_loss)
                            self._history_valid_loss_trace.append(self._total_train_traces)
                            last_validation_trace = trace - 1

                            if distributed_world_size > 1:
                                self._distributed_update_train_loss(loss, distributed_world_size)
                                self._distributed_update_valid_loss(valid_loss, distributed_world_size)

                    if (distributed_world_size > 1) and (iteration % distributed_loss_update_every == 0):
                        self._distributed_update_train_loss(loss, distributed_world_size)

                    if (distributed_rank == 0) and (save_file_name_prefix is not None):
                        if time.time() - last_auto_save_time > save_every_sec:
                            last_auto_save_time = time.time()
                            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
                            print('\rSaving to disk...  ', end='\r')
                            self._save(file_name)

                    print_line = '{} | {} | {} | {} | {} | {} | {} | {}'.format(total_training_seconds_str, epoch_str, total_train_traces_str, loss_initial_str, loss_min_str, loss_str, time_since_loss_min_str, traces_per_second_str)
                    max_print_line_len = max(len(print_line), max_print_line_len)
                    print(print_line.ljust(max_print_line_len), end='\r')
                    sys.stdout.flush()
                    if stop:
                        break
                iteration += 1

        print()
        if (distributed_rank == 0) and (save_file_name_prefix is not None):
            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
            print('\rSaving to disk...  ', end='\r')
            self._save(file_name)
