import torch
import time
import sys
import os
import math
import random
from termcolor import colored

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, Optimizer, AddressDictionary
from .nn import InferenceNetwork as InferenceNetworkBase
from .nn import OnlineDataset, OfflineDataset, InferenceNetworkFeedForward, InferenceNetworkLSTM
from .remote import ModelServer
import horovod.torch as hvd

class Model():
    def __init__(self, name='Unnamed pyprob model', address_dict_file_name=None):
        super().__init__()
        self.name = name
        self._inference_network = None
        if address_dict_file_name is None:
            self._address_dictionary = None
        else:
            self._address_dictionary = AddressDictionary(address_dict_file_name)

    def forward(self):
        raise NotImplementedError()

    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, *args, **kwargs):
        state._init_traces(self.forward, trace_mode, prior_inflation, inference_engine, inference_network, observe, metropolis_hastings_trace, self._address_dictionary)
        while True:
            state._begin_trace()
            result = self.forward(*args, **kwargs)
            trace = state._end_trace(result)
            yield trace

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, silent=False, observe=None, file_name=None, *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, *args, **kwargs)
        traces = Empirical(file_name=file_name)
        time_start = time.time()
        if (util._verbosity > 1) and not silent:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
            prev_duration = 0
        for i in range(num_traces):
            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()
            trace = next(generator)
            if trace_mode == TraceMode.PRIOR:
                log_weight = 1.
            else:
                log_weight = trace.log_importance_weight
            if map_func is not None:
                trace = map_func(trace)
            traces.add(trace, log_weight)
        if (util._verbosity > 1) and not silent:
            print()
        traces.finalize()
        return traces

    def prior_traces(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=None, file_name=None, *args, **kwargs):
        prior = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, *args, **kwargs)
        prior.rename('Prior, traces: {:,}'.format(prior.length))
        return prior

    def prior_distribution(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=lambda trace: trace.result, file_name=None, *args, **kwargs):
        return self.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, *args, **kwargs)

    def posterior_traces(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=None, observe=None, file_name=None, *args, **kwargs):
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
            posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, map_func=map_func, observe=observe, file_name=file_name, *args, **kwargs)
            posterior.rename('Posterior, IS, traces: {:,}, ESS: {:,.2f}'.format(posterior.length, posterior.effective_sample_size))
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            if self._inference_network is None:
                raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
            self._inference_network.eval()
            with torch.no_grad():
                posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=observe, file_name=file_name, *args, **kwargs)
            posterior.rename('Posterior, IC, traces: {:,}, train. traces: {:,}, ESS: {:,.2f}'.format(posterior.length, self._inference_network._total_train_traces, posterior.effective_sample_size))
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
            posterior = Empirical(file_name=file_name)
            if initial_trace is None:
                current_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observe=observe, *args, **kwargs))
            else:
                current_trace = initial_trace

            time_start = time.time()
            traces_accepted = 0
            samples_reused = 0
            samples_all = 0
            if util._verbosity > 1:
                len_str_num_traces = len(str(num_traces))
                print('Time spent  | Time remain.| Progress             | {} | Accepted|Smp reuse| Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
                prev_duration = 0
            for i in range(num_traces):
                if util._verbosity > 1:
                    duration = time.time() - time_start
                    if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                        prev_duration = duration
                        traces_per_second = (i + 1) / duration
                        print('{} | {} | {} | {}/{} | {} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:,.2f}%'.format(100 * (traces_accepted / (i + 1))).rjust(7), '{:,.2f}%'.format(100 * samples_reused / max(1, samples_all)).rjust(7), traces_per_second), end='\r')
                        sys.stdout.flush()
                candidate_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, metropolis_hastings_trace=current_trace, observe=observe, *args, **kwargs))
                log_acceptance_ratio = math.log(current_trace.length_controlled) - math.log(candidate_trace.length_controlled) + candidate_trace.log_prob_observed - current_trace.log_prob_observed
                for variable in candidate_trace.variables_controlled:
                    if variable.reused:
                        log_acceptance_ratio += torch.sum(variable.log_prob)
                        log_acceptance_ratio -= torch.sum(current_trace.variables_dict_address[variable.address].log_prob)
                        samples_reused += 1
                samples_all += candidate_trace.length_controlled

                if state._metropolis_hastings_site_transition_log_prob is None:
                    print(colored('Warning: trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls', 'red', attrs=['bold']))
                else:
                    log_acceptance_ratio += torch.sum(state._metropolis_hastings_site_transition_log_prob)

                # print(log_acceptance_ratio)
                if math.log(random.random()) < float(log_acceptance_ratio):
                    traces_accepted += 1
                    current_trace = candidate_trace
                if map_func is not None:
                    posterior.add(map_func(current_trace))
                else:
                    posterior.add(current_trace)
            if util._verbosity > 1:
                print()

            posterior.finalize()
            posterior.rename('Posterior, {}, traces: {:,}, accepted: {:,.2f}%, sample reuse: {:,.2f}%'.format('LMH' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'RMH', posterior.length, 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all))

        return posterior

    def posterior_distribution(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=lambda trace: trace.result, observe=None, file_name=None, *args, **kwargs):
        return self.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=map_func, observe=observe, file_name=file_name, *args, **kwargs)

    def learn_inference_network(self, num_traces, inference_network=InferenceNetwork.FEEDFORWARD, prior_inflation=PriorInflation.DISABLED, dataset_dir=None, dataset_valid_dir=None, observe_embeddings={}, batch_size=64, valid_size=None, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate=0.001, momentum=0.9, weight_decay=0., save_file_name_prefix=None, save_every_sec=600, pre_generate_layers=True, distributed_backend=None, dataloader_offline_num_workers=0, stop_with_bad_loss=True):
        if dataset_dir is None:
            dataset = OnlineDataset(model=self, prior_inflation=prior_inflation)
        else:
            dataset = OfflineDataset(dataset_dir=dataset_dir)

        dataset_valid = None
        if dataset_valid_dir is None:
            if valid_size is not None:
                dataset_valid = OnlineDataset(model=self, length=valid_size, prior_inflation=prior_inflation)
        else:
            dataset_valid = OfflineDataset(dataset_dir=dataset_valid_dir)

        if self._inference_network is None:
            print('Creating new inference network...')
            if inference_network == InferenceNetwork.FEEDFORWARD:
                self._inference_network = InferenceNetworkFeedForward(model=self, observe_embeddings=observe_embeddings)
            elif inference_network == InferenceNetwork.LSTM:
                self._inference_network = InferenceNetworkLSTM(model=self, observe_embeddings=observe_embeddings)
            else:
                raise ValueError('Unknown inference_network: {}'.format(inference_network))
            if pre_generate_layers:
                if dataset_valid_dir is not None:
                    self._inference_network._pre_generate_layers(dataset_valid, save_file_name_prefix=save_file_name_prefix)
                if dataset_dir is not None:
                    self._inference_network._pre_generate_layers(dataset, save_file_name_prefix=save_file_name_prefix)
        else:
            print('Continuing to train existing inference network...')
            print('Total number of parameters: {:,}'.format(self._inference_network._history_num_params[-1]))

        self._inference_network.to(device=util._device)
        
        Enable_autograd_profile=True
        Enable_Cprofile=True
        profile_dir='/global/cscratch1/sd/leishao/etalumis_data_dec7_2018_small/'
        if Enable_autograd_profile:
            if Enable_Cprofile:
                import cProfile
                po=cProfile.Profile()
                with torch.autograd.profiler.profile() as prof:
                    po.runcall(self._inference_network.optimize,num_traces,dataset,dataset_valid, batch_size, valid_every, optimizer_type,learning_rate,momentum,weight_decay,save_file_name_prefix,save_every_sec,distributed_backend,dataloader_offline_num_workers, stop_with_bad_loss)
                prof.export_chrome_trace(profile_dir + 'HSW_hvd_0115_'+str(hvd.rank())+'outof'+str(hvd.size()) + '.json')
                po.dump_stats(profile_dir + 'optimize0115_256tracesperrank.dat.hvd'+str(hvd.rank())+'outof'+str(hvd.size()))                                   
            else:
                with torch.autograd.profiler.profile() as prof:
                    self._inference_network.optimize(num_traces=num_traces, dataset=dataset, dataset_valid=dataset_valid, batch_size=batch_size, valid_every=valid_every, optimizer_type=optimizer_type, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, save_file_name_prefix=save_file_name_prefix, save_every_sec=save_every_sec, distributed_backend=distributed_backend, dataloader_offline_num_workers=dataloader_offline_num_workers,stop_with_bad_loss=stop_with_bad_loss)
                #prof.export_chrome_trace('/global/cscratch1/sd/leishao/etalumis_data_dec7_2018_small/newcode_1211_HSW_256traces_bz16pt_omp32_1nodepertask_1node.json')
                prof.export_chrome_trace(profile_dir + 'HSW_hvd_0115_'+str(hvd.rank())+'outof'+str(hvd.size()) + '.json')
        else:
            if Enable_Cprofile:
                import cProfile
                po=cProfile.Profile()
                po.runcall(self._inference_network.optimize,num_traces,dataset,dataset_valid, batch_size, valid_every, optimizer_type,learning_rate,momentum,weight_decay,save_file_name_prefix,save_every_sec,distributed_backend,dataloader_offline_num_workers, stop_with_bad_loss)
                po.dump_stats(profile_dir + 'optimize0115_256tracesperrank.dat.hvd'+str(hvd.rank())+'outof'+str(hvd.size()))
            else:
                self._inference_network.optimize(num_traces=num_traces, dataset=dataset, dataset_valid=dataset_valid, batch_size=batch_size, valid_every=valid_every, optimizer_type=optimizer_type, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, save_file_name_prefix=save_file_name_prefix, save_every_sec=save_every_sec, distributed_backend=distributed_backend, dataloader_offline_num_workers=dataloader_offline_num_workers, stop_with_bad_loss=stop_with_bad_loss)

    def save_inference_network(self, file_name):
        if self._inference_network is None:
            raise RuntimeError('The model has no trained inference network.')
        self._inference_network._save(file_name)

    def load_inference_network(self, file_name):
        self._inference_network = InferenceNetworkBase._load(file_name)
        # The following is due to a temporary hack related with https://github.com/pytorch/pytorch/issues/9981 and can be deprecated by using dill as pickler with torch > 0.4.1
        self._inference_network._model = self

    def save_dataset(self, dataset_dir, num_traces, num_traces_per_file, prior_inflation=PriorInflation.DISABLED, *args, **kwargs):
        if not os.path.exists(dataset_dir):
            print('Directory does not exist, creating: {}'.format(dataset_dir))
            os.makedirs(dataset_dir)
        dataset = OnlineDataset(self, None, prior_inflation=prior_inflation)
        dataset.save_dataset(dataset_dir=dataset_dir, num_traces=num_traces, num_traces_per_file=num_traces_per_file, *args, **kwargs)


class RemoteModel(Model):
    def __init__(self, server_address='tcp://127.0.0.1:5555', *args, **kwargs):
        self._server_address = server_address
        self._model_server = None
        super().__init__(*args, **kwargs)

    def close(self):
        if self._model_server is not None:
            self._model_server.close()
        super().close()

    def forward(self):
        if self._model_server is None:
            self._model_server = ModelServer(self._server_address)
            self.name = '{} running on {}'.format(self._model_server.model_name, self._model_server.system_name)

        return self._model_server.forward()
