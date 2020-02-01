import os
import shutil
import uuid
import tempfile
import tarfile
from threading import Thread
import copy
import torch
import numpy as np
import torch.nn as nn
from termcolor import colored

from . import EmbeddingFeedForward, InferenceNetwork, SurrogateAddressTransition, \
    SurrogateNormal, SurrogateCategorical, SurrogateUniform, \
    SurrogateGamma, SurrogateBeta
from .. import util, state
from ..distributions import Normal, Uniform, Categorical, Poisson
from ..trace import Variable, Trace
from .. import __version__


class SurrogateNetworkLSTM(InferenceNetwork):

    def __init__(self, lstm_dim=512, lstm_depth=1, sample_embedding_dim=4,
                 address_embedding_dim=64, distribution_type_embedding_dim=8,
                 batch_norm=False, variable_embeddings={}, *args, **kwargs):
        super().__init__(network_type='SurrogateNetworkLSTM', *args, **kwargs)
        self._layers_sample_embedding = nn.ModuleDict()
        self._layers_address_embedding = nn.ParameterDict()

        self._layers_distribution_type_embedding = nn.ParameterDict()
        self._layers_lstm = None
        self._lstm_input_dim = None
        self._lstm_dim = lstm_dim
        self._lstm_depth = lstm_depth
        self._infer_lstm_state = None
        self._sample_embedding_dim = sample_embedding_dim
        self._address_embedding_dim = address_embedding_dim
        self._distribution_type_embedding_dim = distribution_type_embedding_dim
        self._batch_norm = batch_norm
        self._variable_embeddings = variable_embeddings

        # Surrogate attributes
        self._layers_address_transitions = nn.ModuleDict()
        self._layers_surrogate_distributions = nn.ModuleDict()

        # normalizers
        self._normalizer_bias = nn.ParameterDict()
        self._normalizer_std = nn.ParameterDict()
        self._n_keys_normalizers = 0

        self._tagged_addresses = []
        self._control_addresses = {}
        self._address_base = {}
        self._address_to_name = {}
        self._trace_hashes = set([])
        self._address_path = None

    def _init_layers(self):
        self._lstm_input_dim = self._sample_embedding_dim + 2 * (self._address_embedding_dim + self._distribution_type_embedding_dim)
        self._layers_lstm = nn.LSTM(self._lstm_input_dim, self._lstm_dim, self._lstm_depth)
        self._layers_lstm.to(device=self._device)

    def _init_layers_observe_embedding(self, observe_embedding, example_trace):
        pass

    def _embed_observe(self):
        raise NotImplementedError()

    def _pre_generate_layers(self, dataset):
        raise NotImplementedError()

    def _polymorph(self, batch):
        layers_changed = False
        if self._n_keys_normalizers < len(self._normalizer_bias.keys()) + len(self._normalizer_std.keys()):
            self._n_keys_normalizers = len(self._normalizer_bias.keys()) + len(self._normalizer_std.keys())
            layers_changed = True

        for meta_data, torch_data in batch.sub_batches:
            old_address = "__init"
            for time_step in np.sort(meta_data['latent_time_steps']+meta_data['observed_time_steps']):
                address = meta_data['addresses'][time_step]
                distribution_name = meta_data['distribution_names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                name = meta_data['names'][time_step]
                dist_constants = meta_data['distribution_constants'][time_step]

                if address not in self._address_base or address not in self._address_to_name:
                    self._address_base[address] = "__".join(address.split("__")[:-2])
                    self._address_to_name[address] = name

                if address not in self._layers_address_embedding:
                    emb = nn.Parameter(torch.zeros(self._address_embedding_dim).normal_().to(device=self._device))
                    self._layers_address_embedding[address] = emb

                if distribution_name not in self._layers_distribution_type_embedding:
                    emb = nn.Parameter(torch.zeros(self._distribution_type_embedding_dim).normal_().to(device=self._device))
                    self._layers_distribution_type_embedding[distribution_name] = emb

                if old_address not in self._layers_address_transitions:
                    if not old_address == "__init":
                        self._layers_address_transitions[old_address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim,
                                                                                                   address).to(device=self._device)
                    else:
                        self._layers_address_transitions[old_address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim,
                                                                                                   address,
                                                                                                   first_address=True).to(device=self._device)
                        layers_changed = True
                else:
                    if address not in self._layers_address_transitions[old_address]._address_to_class:
                        self._layers_address_transitions[old_address].add_address_transition(address)
                        layers_changed = True
                old_address = address
                if address not in self._layers_surrogate_distributions:
                    variable_shape = torch_data[time_step]['values'][0].shape
                    if name in self._variable_embeddings:
                        var_embedding = self._variable_embeddings[name]
                    else:
                        var_embedding = {'num_layers': 2,
                                         'hidden_dim': None}
                    if distribution_name == 'Normal':
                        distribution = torch_data[time_step]['distribution']
                        # ignore batch dim for the shapes
                        loc_shape, scale_shape = distribution.loc_shape[1:], distribution.scale_shape[1:]
                        surrogate_distribution = SurrogateNormal(self._lstm_dim,
                                                                 loc_shape,
                                                                 scale_shape,
                                                                 dist_constants,
                                                                 **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Uniform':
                        surrogate_distribution = SurrogateUniform(self._lstm_dim, variable_shape,
                                                                  dist_constants,
                                                                  **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Gamma':
                        distribution = torch_data[time_step]['distribution']
                        shape_shape, rate_shape = distribution.shape_shape[1:], distribution.rate_shape[1:]
                        surrogate_distribution = SurrogateGamma(self._lstm_dim, shape_shape, rate_shape,
                                                                dist_constants,
                                                                **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Beta':
                        distribution = torch_data[time_step]['distribution']
                        concentration1_shape, concentration0_shape = distribution.concentration1_shape[1:], distribution.concentration0_shape[1:]
                        surrogate_distribution = SurrogateBeta(self._lstm_dim,
                                                               concentration1_shape,
                                                               concentration0_shape,
                                                               dist_constants,
                                                               **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Poisson':
                        surrogate_distribution = SurrogatePoisson(self._lstm_dim,
                                                                  variable_shape,
                                                                  dist_constants,
                                                                  **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)
                    elif distribution_name == 'Categorical':
                        distribution = torch_data[time_step]['distribution']
                        surrogate_distribution = SurrogateCategorical(self._lstm_dim,
                                                                      distribution.num_categories,
                                                                      dist_constants,
                                                                      **var_embedding)
                        sample_embedding_layer = EmbeddingFeedForward(variable_shape,
                                                                      self._sample_embedding_dim,
                                                                      num_layers=1)

                    surrogoate_distribution = surrogate_distribution.to(device=self._device)
                    sample_embedding_layer = sample_embedding_layer.to(device=self._device)
                    self._layers_sample_embedding[address] = sample_embedding_layer
                    self._layers_surrogate_distributions[address] = surrogate_distribution
                    layers_changed = True
                    print('New layers, address: {}, distribution: {}'.format(util.truncate_str(address), distribution_name))

            # add final address transition that ends the trace
            if address not in self._layers_address_transitions:
                self._layers_address_transitions[address] = SurrogateAddressTransition(self._lstm_dim + self._sample_embedding_dim,
                                                                                       None, last_address=True).to(device=self._device)

        if layers_changed:
            num_params = sum(p.numel() for p in self.parameters())
            print('Total addresses: {:,}, distribution types: {:,}, parameters: {:,}'.format(len(self._layers_address_embedding), len(self._layers_distribution_type_embedding), num_params))
            self._history_num_params.append(num_params)
            self._history_num_params_trace.append(self._total_train_traces)
        return layers_changed

    def run_lstm_step(self, variable, prev_variable=None):
        success = True
        if prev_variable is None:
            # First time step
            prev_sample_embedding = torch.zeros(1, self._sample_embedding_dim).to(device=self._device)
            prev_address_embedding = torch.zeros(self._address_embedding_dim).to(device=self._device)
            prev_distribution_type_embedding = torch.zeros(self._distribution_type_embedding_dim).to(device=self._device)
            h0 = torch.zeros(self._lstm_depth, 1, self._lstm_dim).to(device=self._device)
            c0 = torch.zeros(self._lstm_depth, 1, self._lstm_dim).to(device=self._device)
            self._lstm_state = (h0, c0)
        else:
            prev_address = prev_variable.address
            prev_distribution = prev_variable.distribution
            prev_value = prev_variable.value.to(device=self._device)
            if prev_value.dim() == 0:
                prev_value = prev_value.unsqueeze(0)
            if prev_address in self._layers_address_embedding:
                prev_value = (prev_value - self._normalizer_bias[prev_address])/self._normalizer_std[prev_address]
                prev_sample_embedding = self._layers_sample_embedding[prev_address](prev_value.float())
                prev_address_embedding = self._layers_address_embedding[prev_address]
                prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution.name]
            else:
                print('Warning: address of previous variable unknown by surrogate network: {}'.format(prev_address))
                success = False

        current_address = variable.address
        current_distribution = variable.distribution
        if current_address in self._layers_address_embedding:
            current_address_embedding = self._layers_address_embedding[current_address]
            current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution.name]
        else:
            print('Warning: address of current variable unknown by surrogate network: {}'.format(current_address))
            success = False

        if success:
            t = torch.cat([prev_sample_embedding[0],
                           prev_distribution_type_embedding,
                           prev_address_embedding,
                           current_distribution_type_embedding,
                           current_address_embedding]).unsqueeze(0)
            lstm_input = t.unsqueeze(0)
            lstm_output, self._lstm_state = self._layers_lstm(lstm_input, self._lstm_state)
            return success, lstm_output
        else:
            return success, None

    def _loss(self, batch):
        batch_loss = 0.
        for meta_data, torch_data in batch.sub_batches:
            trace_hash = meta_data['trace_hash']
            if trace_hash not in self._trace_hashes:
                self._trace_hashes.add(trace_hash)
            sub_batch_length = torch_data[0]['values'].size(0)
            sub_batch_loss = 0.
            # print('sub_batch_length', sub_batch_length, 'example_trace_length_controlled', example_trace.length_controlled, '  ')

            # Construct LSTM input sequence for the whole trace length of sub_batch
            lstm_input = []
            for time_step in np.sort(meta_data['latent_time_steps']+meta_data['observed_time_steps']):
                current_address = meta_data['addresses'][time_step]
                current_distribution_name = meta_data['distribution_names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                current_name = meta_data['names'][time_step]

                if (current_address not in self._layers_address_embedding) and (current_address not in self._layers_surrogate_distributions):
                    print(colored('Address unknown by surrogate network: {}'.format(current_address), 'red', attrs=['bold']))
                    return False, 0
                current_address_embedding = self._layers_address_embedding[current_address]
                current_distribution_type_embedding = self._layers_distribution_type_embedding[current_distribution_name]


                if time_step == 0:
                    prev_sample_embedding = torch.zeros(sub_batch_length, self._sample_embedding_dim).to(device=self._device)
                    prev_address_embedding = torch.zeros(1, self._address_embedding_dim).to(device=self._device)
                    prev_distribution_type_embedding = torch.zeros(1, self._distribution_type_embedding_dim).to(device=self._device)
                else:
                    prev_address = meta_data['addresses'][time_step-1]
                    prev_distribution_name = meta_data['distribution_names'][time_step-1]
                    if prev_address not in self._layers_address_embedding:
                        print(colored('Address unknown by surrogate network: {}'.format(prev_address), 'red', attrs=['bold']))
                        return False, 0
                    smp = torch_data[time_step-1]['values'].to(device=self._device)

                    if prev_address not in self._normalizer_bias:
                        self._normalizer_bias.update({prev_address: nn.Parameter(smp.mean(dim=0), requires_grad=False)})
                    if prev_address not in self._normalizer_std:
                        # TODO consider what happens if batchnorm is small - ie 1 !!
                        tmp_std = smp.std(dim=0)
                        mask = (tmp_std == 0) | torch.isnan(tmp_std)
                        tmp_std[mask] = 1
                        self._normalizer_std.update({prev_address: nn.Parameter(tmp_std, requires_grad=False)})

                    # TODO BE CAREFUL - THIS IS A SUB MINIBATCH. THE NORMALIZATION COULD CHANGE IMMENSELY ACROSS SUBMINIBATCH
                    # CONSIDER USING RUNNING MEANS, ETC...
                    # MAYBE NORMALIZE FOR EACH TRACETYPE
                    smp = (smp - self._normalizer_bias[prev_address])/self._normalizer_std[prev_address]
                    prev_sample_embedding = self._layers_sample_embedding[prev_address](smp)
                    prev_address_embedding = self._layers_address_embedding[prev_address]
                    prev_distribution_type_embedding = self._layers_distribution_type_embedding[prev_distribution_name]

                # concat to size batch_size x *
                t = torch.cat([prev_sample_embedding,
                               prev_distribution_type_embedding.repeat(sub_batch_length, 1),
                               prev_address_embedding.repeat(sub_batch_length, 1),
                               current_distribution_type_embedding.repeat(sub_batch_length, 1),
                               current_address_embedding.repeat(sub_batch_length, 1)], dim=1)
                lstm_input.append(t)

            # Execute LSTM in a single operation on the whole input sequence
            lstm_input = torch.stack(lstm_input, dim=0)
            h0 = torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim).to(device=self._device)
            c0 = torch.zeros(self._lstm_depth, sub_batch_length, self._lstm_dim).to(device=self._device)
            lstm_output, _ = self._layers_lstm(lstm_input, (h0, c0))

            # surrogate loss
            surrogate_loss = 0.
            trace_length = len(meta_data['latent_time_steps']+meta_data['observed_time_steps'])
            for time_step in np.sort(meta_data['latent_time_steps']+meta_data['observed_time_steps']):
                address = meta_data['addresses'][time_step]
                current_name = meta_data['names'][time_step]
                current_controlled = meta_data['controls'][time_step]
                if current_controlled:
                    self._control_addresses[address] = True
                else:
                    self._control_addresses[address] = False
                proposal_input = lstm_output[time_step]
                if time_step < trace_length - 1:
                    next_addresses = [meta_data['addresses'][time_step+1]]*sub_batch_length
                else:
                    next_addresses = ["__end"]*sub_batch_length
                variable_dist = torch_data[time_step]['distribution'].to(device=self._device)
                address_transition_layer = self._layers_address_transitions[address]
                surrogate_distribution_layer = self._layers_surrogate_distributions[address]

                # only consider loss and training of the address transition if we are not at the end of trace
                if time_step < trace_length - 1:
                    values = torch_data[time_step]['values'].to(device=self._device)
                    values = (values - self._normalizer_bias[address])/self._normalizer_std[address]
                    sample_embedding = self._layers_sample_embedding[address](values)
                    address_transition_input = torch.cat([proposal_input, sample_embedding], dim=1)

                    _ = address_transition_layer(address_transition_input)
                    surrogate_loss += torch.sum(address_transition_layer._loss(next_addresses))

                _ = surrogate_distribution_layer(proposal_input)
                values = torch_data[time_step]['values'].to(device=self._device)
                surrogate_loss += torch.sum(surrogate_distribution_layer._loss(values))

            batch_loss += sub_batch_loss + surrogate_loss
        return True, batch_loss / batch.size

    def get_surrogate_forward(self, original_forward=lambda x: x):
        self._original_forward = original_forward
        return self.forward

    def fix_address_transitions(self, address_path):
        self._address_path = address_path

    def forward(self, *args, **kwargs):
        """
        !! NOT TO BE USED AS A PyTorch FORWARD METHOD !!

        Rewrite the forward function otherwise specified by the user.

        This forward function uses the surrogate model as joint distribution

        """
        # sample initial address

        with torch.no_grad(): # DO NOT ADD THIS NETWORKS PARAMETERS TO GRADIENT COMPUTATION GRAPH
            if not self._address_path:
                address = self._layers_address_transitions["__init"](None, batch_size=1).sample()
            else:
                address = self._address_path[0]
            prev_variable = None
            time_step = 1
            prev_address = address
            while address != "__end":
                surrogate_dist = self._layers_surrogate_distributions[address]
                current_variable = Variable(distribution=surrogate_dist.dist_type,
                                            address=address,
                                            value=None)
                # normalization happens in run_lstm_step
                _, lstm_output = self.run_lstm_step(current_variable, prev_variable)
                address_dist = self._layers_address_transitions[address]
                lstm_output = lstm_output.squeeze(0) # remove sequence dim
                dist = surrogate_dist(lstm_output, no_batch=True) # remove batch

                value = state.sample(distribution=dist,
                                     address=self._address_base[address],
                                     name=self._address_to_name[address],
                                     control=self._control_addresses[address])

                if address in self._tagged_addresses:
                    state.tag(value, address=self._address_base[address])

                prev_variable = Variable(distribution=surrogate_dist.dist_type,
                                         address=address, value=value.unsqueeze(0))

                # normalize value for the address transitions
                if address in self._normalizer_bias and address in self._normalizer_std:
                    tmp = (value - self._normalizer_bias[address])/self._normalizer_std[address]
                sample_embedding = self._layers_sample_embedding[address](tmp.unsqueeze(0)) # add batch again!

                address_transition_input = torch.cat([lstm_output, sample_embedding], dim=1)
                a_dist = address_dist(address_transition_input)
                if not self._address_path:
                    address = a_dist.sample()
                else:
                    address = self._address_path[time_step]

                time_step += 1

                if address == "__unknown":
                    print(colored(f"Warning: sampled unknown address at address: {prev_address}",
                                  'red', attrs=['bold']))
                    print(colored(f"These are the address probabilities (THE LAST IS UNKNOWN!!): \n\t {torch.exp(a_dist._logits)}",
                                  'red', attrs=['bold']))
                    # if an address is unknown default to the simulator
                    # by resetting the _current_trace
                    state._current_trace = Trace()
                    self._original_forward(*args, **kwargs)
                    break

                prev_address = address

            # TODO a better data structure than a set?
            # Is this even necessary assuming we trust the model?
            # Combinatorical issues...
            #trace_hash = self._trace_hashing(state._current_trace)
            #if trace_hash not in self._trace_hashes:
                # if a trace was not seen during training default to simulator
            #    state._current_trace = Trace()
            #    self._original_forward(*args, **kwargs)

            return None

    def _save(self, file_name):
        self._modified = util.get_time_str()
        self._updates += 1

        data = {}
        data['pyprob_version'] = __version__
        data['torch_version'] = torch.__version__
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        data['surrogate_network'] = copy.copy(self)
        data['surrogate_network']._model = None
        data['surrogate_network']._optimizer = None
        if self._optimizer is None:
            data['surrogate_network']._optimizer_state = None
        else:
        #    self._create_optimizer()
            data['surrogate_network']._optimizer_state = self._optimizer.state_dict()
        data['surrogate_network']._learning_rate_scheduler = None
        if self._learning_rate_scheduler is None:
            data['surrogate_network']._learning_rate_scheduler_state = None
        else:
            data['surrogate_network']._learning_rate_scheduler_state = self._learning_rate_scheduler.state_dict()

        def thread_save():
            tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
            tmp_file_name = os.path.join(tmp_dir, 'pyprob_surrogate_network')
            torch.save(data, tmp_file_name)
            tar = tarfile.open(file_name, 'w:gz', compresslevel=2)
            tar.add(tmp_file_name, arcname='pyprob_surrogate_network')
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
            tmp_file = os.path.join(tmp_dir, 'pyprob_surrogate_network')
            tar.extract('pyprob_surrogate_network', tmp_dir)
            tar.close()
            if util._cuda_enabled:
                data = torch.load(tmp_file)
            else:
                data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(e)
            raise RuntimeError('Cannot load surrogate network.')

        if data['pyprob_version'] != __version__:
            print(colored('Warning: different pyprob versions (loaded network: {}, current system: {})'.format(data['pyprob_version'], __version__), 'red', attrs=['bold']))
        if data['torch_version'] != torch.__version__:
            print(colored('Warning: different PyTorch versions (loaded network: {}, current system: {})'.format(data['torch_version'], torch.__version__), 'red', attrs=['bold']))

        ret = data['surrogate_network']
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
