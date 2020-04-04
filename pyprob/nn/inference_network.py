import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
import math
from threading import Thread
from termcolor import colored

from . import Batch, OfflineDataset, TraceBatchSampler, DistributedTraceBatchSampler, EmbeddingFeedForward, EmbeddingCNN2D5C, EmbeddingCNN3D5C
from .optimizer_larc import LARC
from .. import __version__, util, Optimizer, LearningRateScheduler, ObserveEmbedding


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
        self._optimizer_type = None
        self._optimizer_state = None
        self._momentum = None
        self._weight_decay = None
        self._learning_rate_scheduler = None
        self._learning_rate_scheduler_type = None
        self._learning_rate_scheduler_state = None

        self._total_train_seconds = 0
        self._total_train_traces = 0
        self._total_train_traces_end = None
        self._total_train_iterations = 0
        self._learning_rate_init = None
        self._learning_rate_end = None
        self._loss_init = None
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
        self._distributed_history_train_loss = []
        self._distributed_history_train_loss_trace = []
        self._distributed_history_valid_loss = []
        self._distributed_history_valid_loss_trace = []
        self._modified = util.get_time_str()
        self._updates = 0
        self._on_cuda = False
        self._device = torch.device('cpu')
        self._learning_rate = None
        self._momentum = None
        self._batch_size = None
        self._distributed_backend = None
        self._distributed_world_size = None
        self._network_type = network_type

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
                print('Observable {}: reshape to {}.'.format(name, input_shape))
            else:
                input_shape = variable.value.size()
                print('Observable {}: reshape not specified, using shape {}.'.format(name, input_shape))
            if 'dim' in value:
                output_shape = torch.Size([value['dim']])
                print('Observable {}: using embedding dim {}.'.format(name, output_shape))
            else:
                print('Observable {}: embedding dim not specified, using the default 256.'.format(name))
                output_shape = torch.Size([256])
            if 'embedding' in value:
                embedding = value['embedding']
                print('Observable {}: using observe embedding {}.'.format(name, embedding))
            else:
                print('Observable {}: observe embedding not specified, using the default FEEDFORWARD.'.format(name))
                embedding = ObserveEmbedding.FEEDFORWARD
            if embedding == ObserveEmbedding.FEEDFORWARD:
                if 'depth' in value:
                    depth = value['depth']
                    print('Observable {}: using embedding depth {}.'.format(name, depth))
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
        if self._optimizer is None:
            data['inference_network']._optimizer_state = None
        else:
            data['inference_network']._optimizer_state = self._optimizer.state_dict()
        data['inference_network']._learning_rate_scheduler = None
        if self._learning_rate_scheduler is None:
            data['inference_network']._learning_rate_scheduler_state = None
        else:
            data['inference_network']._learning_rate_scheduler_state = self._learning_rate_scheduler.state_dict()

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
        except Exception as e:
            print(e)
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

        # For compatibility loading NNs saved before 0.13.2.dev2
        if not hasattr(ret, '_distributed_train_loss'):
            ret._distributed_train_loss = util.to_tensor(0.)
        if not hasattr(ret, '_distributed_valid_loss'):
            ret._distributed_valid_loss = util.to_tensor(0.)
        if not hasattr(ret, '_distributed_history_train_loss'):
            ret._distributed_history_train_loss = []
        if not hasattr(ret, '_distributed_history_train_loss_trace'):
            ret._distributed_history_train_loss_trace = []
        if not hasattr(ret, '_distributed_history_valid_loss'):
            ret._distributed_history_valid_loss = []
        if not hasattr(ret, '_distributed_history_valid_loss_trace'):
            ret._distributed_history_valid_loss_trace = []
        if not hasattr(ret, '_optimizer_state'):
            ret._optimizer_state = None
        if not hasattr(ret, '_learning_rate_scheduler_state'):
            ret._learning_rate_scheduler_state = None
        # For compatibility loading NNs saved before 0.13.2.dev5
        if not hasattr(ret, '_total_train_traces_end'):
            ret._total_train_traces_end = None
        # For compatibility loading NNs saved before 0.13.2.dev6
        if not hasattr(ret, '_loss_init'):
            ret._loss_init = None
        if not hasattr(ret, '_learning_rate_init'):
            ret._learning_rate_init = 0
        if not hasattr(ret, '_learning_rate_end'):
            ret._learning_rate_end = 0
        if not hasattr(ret, '_weight_decay'):
            ret._weight_decay = 0
        if not hasattr(ret, '_learning_rate_scheduler_type'):
            ret._learning_rate_scheduler_type = None

        ret._create_optimizer(ret._optimizer_state)
        ret._create_lr_scheduler(ret._learning_rate_scheduler_state)
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: Batch(x))
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

    def _distributed_sync_grad(self, world_size):
        """ all_reduce grads from all ranks """
        # print('Distributed training synchronizing gradients across nodes...')
        # make a local map of all non-zero gradients
        ttmap = util.to_tensor([1 if p.grad is not None else 0 for p in self.parameters()])
        # get the global map of all non-zero gradients
        pytorch_allreduce_supports_list = True
        try:
            dist.all_reduce([ttmap])
        except:
            pytorch_allreduce_supports_list = False
            dist.all_reduce(ttmap)
        gl = []
        for i, param in enumerate(self.parameters()):
            if param.grad is not None:
                gl.append(param.grad.data)
            elif ttmap[i]:
                # someone else had a non-zero grad so make a local zero'd copy
                param.grad = util.to_tensor(torch.zeros_like(param.data))
                gl.append(param.grad.data)

        # reduce all gradients used by at least one rank
        if pytorch_allreduce_supports_list:
            dist.all_reduce(gl)
        else:
            for g in gl:
                dist.all_reduce(g)
        # average them
        for li in gl:
            li /= float(world_size)

    def _distributed_update_train_loss(self, loss, world_size):
        self._distributed_train_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_train_loss)
        self._distributed_train_loss /= float(world_size)
        self._distributed_history_train_loss.append(float(self._distributed_train_loss))
        self._distributed_history_train_loss_trace.append(self._total_train_traces)
        return self._distributed_train_loss

    def _distributed_update_valid_loss(self, loss, world_size):
        self._distributed_valid_loss = util.to_tensor(float(loss))
        dist.all_reduce(self._distributed_valid_loss)
        self._distributed_valid_loss /= float(world_size)
        self._distributed_history_valid_loss.append(float(self._distributed_valid_loss))
        self._distributed_history_valid_loss_trace.append(self._total_train_traces)
        return self._distributed_valid_loss

    def _create_optimizer(self, state_dict=None):
        if self._optimizer_type is None:  # happens when loading pre-generated network
            return
        # print('Creating new optimizer')
        if self._optimizer_type in [Optimizer.ADAM, Optimizer.ADAM_LARC]:
            self._optimizer = optim.Adam(self.parameters(), lr=self._learning_rate_init, weight_decay=self._weight_decay)
        else:  # optimizer_type in [Optimizer.SGD, Optimizer.SGD_LARC]
            self._optimizer = optim.SGD(self.parameters(), lr=self._learning_rate_init, momentum=self._momentum, nesterov=True, weight_decay=self._weight_decay)
        if self._optimizer_type in [Optimizer.ADAM_LARC, Optimizer.SGD_LARC]:
            self._optimizer = LARC(self._optimizer)
        if state_dict is not None:
            # print('Setting optimizer state')
            self._optimizer.load_state_dict(state_dict)

    def _create_lr_scheduler(self, state_dict=None):
        if self._learning_rate_scheduler_type is None:  # happens when loading pre-generated network
            return
        # print('Creating new learning rate scheduler')
        learning_rate_scheduler_type = self._learning_rate_scheduler_type
        iter_end = self._total_train_traces_end
        lr_init = self._learning_rate_init
        lr_end = self._learning_rate_end

        def _poly_decay(iter, power):
            return (lr_init - lr_end) * ((1 - iter/iter_end) ** power) + lr_end

        if self._optimizer is None:
            self._learning_rate_scheduler = None
        elif learning_rate_scheduler_type == LearningRateScheduler.POLY1:
            self._learning_rate_scheduler = lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda iter: _poly_decay(iter, power=1.) / lr_init)
        elif learning_rate_scheduler_type == LearningRateScheduler.POLY2:
            self._learning_rate_scheduler = lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda iter: _poly_decay(iter, power=2.) / lr_init)
        else:
            self._learning_rate_scheduler = None
        if self._learning_rate_scheduler is not None and state_dict is not None:
            # print('Setting learning rate scheduler state')
            self._learning_rate_scheduler.load_state_dict(state_dict)

    def optimize(self, num_traces, dataset, dataset_valid=None, num_traces_end=1e9, batch_size=64, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate_init=0.0001, learning_rate_end=1e-6, learning_rate_scheduler_type=LearningRateScheduler.NONE, momentum=0.9, weight_decay=1e-5, save_file_name_prefix=None, save_every_sec=600, distributed_backend=None, distributed_params_sync_every_iter=10000, distributed_num_buckets=10, dataloader_offline_num_workers=0, stop_with_bad_loss=False, log_file_name=None):
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
            self._distributed_backend = distributed_backend
            self._distributed_world_size = distributed_world_size

        # Training data loader
        if isinstance(dataset, OfflineDataset):
            if distributed_world_size == 1:
                dataloader = DataLoader(dataset, batch_sampler=TraceBatchSampler(dataset, batch_size=batch_size, shuffle_batches=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            else:
                dataloader = DataLoader(dataset, batch_sampler=DistributedTraceBatchSampler(dataset, batch_size=batch_size, num_buckets=distributed_num_buckets, shuffle_batches=True, shuffle_buckets=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=lambda x: Batch(x))

        # Validation data loader
        if dataset_valid is not None:
            if distributed_world_size == 1:
                dataloader_valid = DataLoader(dataset_valid, batch_sampler=TraceBatchSampler(dataset_valid, batch_size=batch_size, shuffle_batches=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            else:
                dataloader_valid = DataLoader(dataset_valid, batch_sampler=DistributedTraceBatchSampler(dataset_valid, batch_size=batch_size, num_buckets=distributed_num_buckets, shuffle_batches=True, shuffle_buckets=True), num_workers=dataloader_offline_num_workers, collate_fn=lambda x: Batch(x))
            if not self._layers_pre_generated:
                for i_batch, batch in enumerate(dataloader_valid):
                    self._polymorph(batch)

        if distributed_world_size > 1:
            util.init_distributed_print(distributed_rank, distributed_world_size, False)
            if distributed_rank == 0:
                print(colored('Distributed synchronous training', 'yellow', attrs=['bold']))
                print(colored('Distributed backend        : {}'.format(distributed_backend), 'yellow', attrs=['bold']))
                print(colored('Distributed world size     : {}'.format(distributed_world_size), 'yellow', attrs=['bold']))
                print(colored('Distributed minibatch size : {} (global effective), {} (per rank)'.format(batch_size * distributed_world_size, batch_size), 'yellow', attrs=['bold']))
                print(colored('Distributed init.learn rate: {} (global), {} (base)'.format(learning_rate_init * math.sqrt(distributed_world_size), learning_rate_init), 'yellow', attrs=['bold']))
                print(colored('Distributed optimizer      : {}'.format(str(optimizer_type)), 'yellow', attrs=['bold']))
                print(colored('Distributed dataset size   : {:,}'.format(len(dataset)), 'yellow', attrs=['bold']))
                print(colored('Distributed num. buckets   : {:,}'.format(len(dataloader.batch_sampler._buckets)), 'yellow', attrs=['bold']))
                # bucket_size = math.ceil((len(dataset) / batch_size) / distributed_num_buckets)
                # print(colored('Distributed bucket size    : {:,} minibatches ({:,} traces)'.format(bucket_size, bucket_size * batch_size), 'yellow', attrs=['bold']))

        self.train()
        prev_total_train_seconds = self._total_train_seconds
        time_start = time.time()
        time_loss_min = time_start
        time_last_batch = time_start
        if valid_every is None:
            valid_every = max(100, num_traces / 1000)
        last_validation_trace = -valid_every + 1
        valid_loss = 0
        if self._optimizer_type is None:
            self._optimizer_type = optimizer_type
        if self._momentum is None:
            self._momentum = momentum
        if self._weight_decay is None:
            self._weight_decay = weight_decay
        if self._learning_rate_scheduler_type is None:
            self._learning_rate_scheduler_type = learning_rate_scheduler_type
        if self._learning_rate_init is None:
            self._learning_rate_init = learning_rate_init * math.sqrt(distributed_world_size)
        if self._learning_rate_end is None:
            self._learning_rate_end = learning_rate_end
        if self._total_train_traces_end is None:
            self._total_train_traces_end = num_traces_end
        epoch = 0
        trace = 0
        stop = False
        print('Train. time | Epoch| Trace     | Init. loss| Min. loss | Curr. loss| T.since min | Learn.rate| Traces/sec')
        max_print_line_len = 0
        loss_min_str = ''
        time_since_loss_min_str = ''
        loss_init_str = '' if self._loss_init is None else '{:+.2e}'.format(self._loss_init)
        if save_every_sec is not None:
            last_auto_save_time = time_start - save_every_sec
        last_print = time_start - util._print_refresh_rate
        if (distributed_rank == 0) and log_file_name is not None:
            log_file = open(log_file_name, mode='w', buffering=1)
            log_file.write('time, iteration, trace, loss, valid_loss, learning_rate, mean_trace_length_controlled, sub_mini_batches, distributed_bucket_id, traces_per_second\n')

        while not stop:
            epoch += 1
            for i_batch, batch in enumerate(dataloader):
                time_batch = time.time()
                # Important, a self._distributed_sync_parameters() needs to happen at the very beginning of a training
                if (distributed_world_size > 1) and (self._total_train_iterations % distributed_params_sync_every_iter == 0):
                    self._distributed_sync_parameters()

                if self._layers_pre_generated:  # and (distributed_world_size > 1):
                    layers_changed = False
                else:
                    layers_changed = self._polymorph(batch)

                if (self._optimizer is None) or layers_changed:
                    self._create_optimizer()
                    self._create_lr_scheduler()

                # print(self._optimizer.state[self._optimizer.param_groups[0]['params'][0]])
                self._optimizer.zero_grad()
                success, loss = self._loss(batch)
                if not success:
                    print(colored('Cannot compute loss, skipping batch. Loss: {}'.format(loss), 'red', attrs=['bold']))
                    if stop_with_bad_loss:
                        return
                else:
                    loss.backward()
                    if distributed_world_size > 1:
                        self._distributed_sync_grad(distributed_world_size)
                    self._optimizer.step()
                    loss = float(loss)
                    if (distributed_world_size > 1):
                        loss = self._distributed_update_train_loss(loss, distributed_world_size)

                    if self._loss_init is None:
                        self._loss_init = loss
                        self._loss_max = loss
                        loss_init_str = '{:+.2e}'.format(self._loss_init)
                    # loss_max_str = '{:+.3e}'.format(self._loss_max)
                    if loss < self._loss_min:
                        self._loss_min = loss
                        loss_str = colored('{:+.2e}'.format(loss), 'green', attrs=['bold'])
                        loss_min_str = colored('{:+.2e}'.format(self._loss_min), 'green', attrs=['bold'])
                        time_loss_min = time_batch
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
                        time_since_loss_min_str = util.days_hours_mins_secs_str(time_batch - time_loss_min)

                    self._loss_previous = loss
                    self._total_train_iterations += 1
                    trace += batch.size * distributed_world_size
                    self._total_train_traces += batch.size * distributed_world_size
                    self._total_train_seconds = prev_total_train_seconds + (time_batch - time_start)
                    self._history_train_loss.append(loss)
                    self._history_train_loss_trace.append(self._total_train_traces)
                    traces_per_second = batch.size * distributed_world_size / (time_batch - time_last_batch)
                    if dataset_valid is not None:
                        if trace - last_validation_trace > valid_every:
                            print('\nComputing validation loss')
                            valid_loss = 0
                            with torch.no_grad():
                                for i_batch, batch in enumerate(dataloader_valid):
                                    _, v = self._loss(batch)
                                    valid_loss += v
                            valid_loss = float(valid_loss) / (len(dataloader_valid) / distributed_world_size)
                            if distributed_world_size > 1:
                                valid_loss = self._distributed_update_valid_loss(valid_loss, distributed_world_size)
                            self._history_valid_loss.append(valid_loss)
                            self._history_valid_loss_trace.append(self._total_train_traces)
                            last_validation_trace = trace - 1

                    if (distributed_rank == 0) and (save_file_name_prefix is not None) and (save_every_sec is not None):
                        if time_batch - last_auto_save_time > save_every_sec:
                            last_auto_save_time = time_batch
                            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
                            print('\rSaving to disk...  ', end='\r')
                            self._save(file_name)

                    time_last_batch = time_batch
                    if trace >= num_traces:
                        print('\nStop condition reached. num_traces: {}'.format(num_traces))
                        stop = True
                    if self._total_train_traces >= self._total_train_traces_end:
                        print(colored('\nStop condition reached. num_traces_end set during network generation: {}'.format(self._total_train_traces_end), 'red', attrs=['bold']))
                        if self._learning_rate_scheduler is not None:
                            print(colored('Warning: continuing training with learning rate scheduler beyond num_traces_end, make sure this is intended'.format(self._total_train_traces_end), 'red', attrs=['bold']))
                        # stop = True

                    if self._learning_rate_scheduler is not None:
                        self._learning_rate_scheduler.step(self._total_train_traces)  # Gives a DeprecationWarning with PyTorch 1.4.0
                    learning_rate_current = self._optimizer.param_groups[0]['lr']
                    learning_rate_current_str = '{:+.2e}'.format(learning_rate_current)

                    if (time_batch - last_print > util._print_refresh_rate) or stop:
                        last_print = time_batch
                        total_training_seconds_str = util.days_hours_mins_secs_str(self._total_train_seconds)
                        epoch_str = '{:4}'.format('{:,}'.format(epoch))
                        total_train_traces_str = '{:9}'.format('{:,}'.format(self._total_train_traces))
                        traces_per_second_str = '{:,.1f}'.format(traces_per_second)

                        print_line = '{} | {} | {} | {} | {} | {} | {} | {} | {} '.format(total_training_seconds_str, epoch_str, total_train_traces_str, loss_init_str, loss_min_str, loss_str, time_since_loss_min_str, learning_rate_current_str, traces_per_second_str)
                        max_print_line_len = max(len(print_line), max_print_line_len)
                        print(print_line.ljust(max_print_line_len), end='\r')
                        sys.stdout.flush()

                    if (distributed_rank == 0) and log_file_name is not None:
                        bucket_id = None
                        if isinstance(dataloader.batch_sampler, DistributedTraceBatchSampler):
                            bucket_id = dataloader.batch_sampler._current_bucket_id
                        log_file.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(self._total_train_seconds, self._total_train_iterations, self._total_train_traces, loss, valid_loss, learning_rate_current, batch.mean_length_controlled, len(batch.sub_batches), bucket_id, traces_per_second))

                    if stop:
                        break

        if (distributed_rank == 0) and log_file_name is not None:
            log_file.close()
        print()
        if (distributed_rank == 0) and (save_file_name_prefix is not None):
            file_name = '{}_{}_traces_{}.network'.format(save_file_name_prefix, util.get_time_stamp(), self._total_train_traces)
            print('\rSaving to disk...  ', end='\r')
            self._save(file_name)
