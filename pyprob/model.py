import torch
import time
import sys
import os
import math
import random
import warnings
import torch.multiprocessing as multiprocessing

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine, InferenceNetwork, Optimizer, LearningRateScheduler, AddressDictionary
from .nn import InferenceNetwork as InferenceNetworkBase
from .nn import OnlineDataset, OfflineDataset, InferenceNetworkFeedForward, InferenceNetworkLSTM
from .remote import ModelServer


def trace_result(trace):
    return trace.result

def trace_id(trace):
    return trace

class Model():
    def __init__(self, name='Unnamed PyProb model', address_dict_file_name=None):
        super().__init__()
        self.name = name
        self._inference_network = None
        if address_dict_file_name is None:
            self._address_dictionary = None
        else:
            self._address_dictionary = AddressDictionary(address_dict_file_name)

    def __repr__(self):
        return 'Model(name:{})'.format(self.name)

    def forward(self):
        raise RuntimeError('Model instances must provide a forward method.')

    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, likelihood_importance=1., *args, **kwargs):
        state._init_traces(func=self.forward, trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, metropolis_hastings_trace=metropolis_hastings_trace, address_dictionary=self._address_dictionary, likelihood_importance=likelihood_importance)
        while True:
            state._begin_trace()
            result = self.forward(*args, **kwargs)
            trace = state._end_trace(result)
            yield trace

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, silent=False, observe=None, file_name=None, likelihood_importance=1., *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, likelihood_importance=likelihood_importance, *args, **kwargs)
        traces = Empirical(file_name=file_name)
        if map_func is None:
            map_func = trace_id
        log_weights = util.to_tensor(torch.zeros(num_traces))
        time_start = time.time()
        if (util._verbosity > 1) and not silent:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1), 'ESS'.ljust(len_str_num_traces+2)))
            prev_duration = 0

        for i in range(num_traces):
            trace = next(generator)
            if trace_mode == TraceMode.PRIOR:
                log_weight = 1.
            else:
                log_weight = trace.log_importance_weight
            if util.has_nan_or_inf(log_weight):
                warnings.warn('Encountered trace with nan, inf, or -inf log_weight. Discarding trace.')
                if i > 0:
                    log_weights[i] = log_weights[-1]
            else:
                traces.add(map_func(trace), log_weight)
                log_weights[i] = log_weight

            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    effective_sample_size = util.effective_sample_size(log_weights[:i+1])
                    if util.has_nan_or_inf(effective_sample_size):
                        effective_sample_size = 0
                    print('{} | {} | {} | {}/{} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:.2f}'.format(effective_sample_size).rjust(len_str_num_traces+2), traces_per_second), end='\r')
                    sys.stdout.flush()

            i += 1
        if (util._verbosity > 1) and not silent:
            print()
        traces.finalize()
        return traces

    def get_trace(self, *args, **kwargs):
        warnings.warn('Model.get_trace will be deprecated in future releases. Use Model.sample instead.')
        return next(self._trace_generator(*args, **kwargs))

    def sample(self, *args, **kwargs):
        return next(self._trace_generator(*args, **kwargs))

    def prior(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=None, file_name=None, likelihood_importance=1., *args, **kwargs):
        prior = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
        prior.rename('Prior, traces: {:,}'.format(prior.length))
        prior.add_metadata(op='prior', num_traces=num_traces, prior_inflation=str(prior_inflation), likelihood_importance=likelihood_importance)
        return prior

    def prior_results(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=trace_result, file_name=None, likelihood_importance=1., *args, **kwargs):
        return self.prior(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)

    def posterior(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=None, observe=None, file_name=None, thinning_steps=None, likelihood_importance=1., *args, **kwargs):
        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
            posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, map_func=map_func, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
            posterior.rename('Posterior, IS, traces: {:,}, ESS: {:,.2f}'.format(posterior.length, posterior.effective_sample_size))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), effective_sample_size=posterior.effective_sample_size, likelihood_importance=likelihood_importance)
        elif inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            if self._inference_network is None:
                raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
            with torch.no_grad():
                posterior = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=self._inference_network, map_func=map_func, observe=observe, file_name=file_name, likelihood_importance=likelihood_importance, *args, **kwargs)
            posterior.rename('Posterior, IC, traces: {:,}, train. traces: {:,}, ESS: {:,.2f}'.format(posterior.length, self._inference_network._total_train_traces, posterior.effective_sample_size))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), effective_sample_size=posterior.effective_sample_size, likelihood_importance=likelihood_importance, train_traces=self._inference_network._total_train_traces)
        else:  # inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
            posterior = Empirical(file_name=file_name)
            if map_func is None:
                map_func = trace_id
            if initial_trace is None:
                initial_trace = next(self._trace_generator(trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, observe=observe, *args, **kwargs))
            if len(initial_trace) == 0:
                raise RuntimeError('Cannot run MCMC inference with empty initial trace. Make sure the model has at least one pyprob.sample statement.')

            current_trace = initial_trace

            time_start = time.time()
            traces_accepted = 0
            samples_reused = 0
            samples_all = 0
            if thinning_steps is None:
                thinning_steps = 1

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
                    warnings.warn('Trace did not hit the Metropolis Hastings site, ensure that the model is deterministic except pyprob.sample calls')
                else:
                    log_acceptance_ratio += torch.sum(state._metropolis_hastings_site_transition_log_prob)

                # print(log_acceptance_ratio)
                if math.log(random.random()) < float(log_acceptance_ratio):
                    traces_accepted += 1
                    current_trace = candidate_trace
                # do thinning
                if i % thinning_steps == 0:
                    posterior.add(map_func(current_trace))

            if util._verbosity > 1:
                print()

            posterior.finalize()
            posterior.rename('Posterior, {}, traces: {:,}{}, accepted: {:,.2f}%, sample reuse: {:,.2f}%'.format('LMH' if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS else 'RMH', posterior.length, '' if thinning_steps == 1 else ' (thinning steps: {:,})'.format(thinning_steps), 100 * (traces_accepted / num_traces), 100 * samples_reused / samples_all))
            posterior.add_metadata(op='posterior', num_traces=num_traces, inference_engine=str(inference_engine), likelihood_importance=likelihood_importance, thinning_steps=thinning_steps, num_traces_accepted=traces_accepted, num_samples_reuised=samples_reused, num_samples=samples_all)
        return posterior

    def posterior_results(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, initial_trace=None, map_func=trace_result, observe=None, file_name=None, thinning_steps=None, *args, **kwargs):
        return self.posterior(num_traces=num_traces, inference_engine=inference_engine, initial_trace=initial_trace, map_func=map_func, observe=observe, file_name=file_name, thinning_steps=thinning_steps, *args, **kwargs)

    def reset_inference_network(self):
        self._inference_network = None

    def learn_inference_network(self, num_traces, num_traces_end=1e9, inference_network=InferenceNetwork.FEEDFORWARD, prior_inflation=PriorInflation.DISABLED, dataset_dir=None, dataset_valid_dir=None, observe_embeddings={}, batch_size=64, valid_size=None, valid_every=None, optimizer_type=Optimizer.ADAM, learning_rate_init=0.001, learning_rate_end=1e-6, learning_rate_scheduler_type=LearningRateScheduler.NONE, momentum=0.9, weight_decay=0., save_file_name_prefix=None, save_every_sec=600, pre_generate_layers=False, distributed_backend=None, distributed_params_sync_every_iter=10000, distributed_num_buckets=None, dataloader_offline_num_workers=0, stop_with_bad_loss=True, log_file_name=None, lstm_dim=512, lstm_depth=1, proposal_mixture_components=10):
        if dataset_dir is None:
            dataset = OnlineDataset(model=self, prior_inflation=prior_inflation)
        else:
            dataset = OfflineDataset(dataset_dir=dataset_dir)

        if dataset_valid_dir is None:
            dataset_valid = None
        else:
            dataset_valid = OfflineDataset(dataset_dir=dataset_valid_dir)

        if self._inference_network is None:
            print('Creating new inference network...')
            if inference_network == InferenceNetwork.FEEDFORWARD:
                self._inference_network = InferenceNetworkFeedForward(model=self, observe_embeddings=observe_embeddings, proposal_mixture_components=proposal_mixture_components)
            elif inference_network == InferenceNetwork.LSTM:
                self._inference_network = InferenceNetworkLSTM(model=self, observe_embeddings=observe_embeddings, lstm_dim=lstm_dim, lstm_depth=lstm_depth, proposal_mixture_components=proposal_mixture_components)
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
        self._inference_network.optimize(num_traces=num_traces, dataset=dataset, dataset_valid=dataset_valid, num_traces_end=num_traces_end, batch_size=batch_size, valid_every=valid_every, optimizer_type=optimizer_type, learning_rate_init=learning_rate_init, learning_rate_end=learning_rate_end, learning_rate_scheduler_type=learning_rate_scheduler_type, momentum=momentum, weight_decay=weight_decay, save_file_name_prefix=save_file_name_prefix, save_every_sec=save_every_sec, distributed_backend=distributed_backend, distributed_params_sync_every_iter=distributed_params_sync_every_iter, distributed_num_buckets=distributed_num_buckets, dataloader_offline_num_workers=dataloader_offline_num_workers, stop_with_bad_loss=stop_with_bad_loss, log_file_name=log_file_name)

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

    def condition(self, criterion, criterion_timeout=1e6):
        return ConditionalModel(self, criterion=criterion, criterion_timeout=criterion_timeout)

    def filter(self, *args, **kwargs):
        warnings.warn('Model.filter will be deprecated in future releases. Use Model.condition instead.')
        return self.condition(*args, **kwargs)

    def parallel(self, num_workers=None):
        return ParallelModel(self, num_workers=num_workers)


class RemoteModel(Model):
    def __init__(self, server_address='tcp://127.0.0.1:5555', before_forward_func=None, after_forward_func=None, *args, **kwargs):
        self._server_address = server_address
        self._model_server = None
        self._before_forward_func = before_forward_func  # Optional mthod to run before each forward call of the remote model (simulator)
        self._after_forward_func = after_forward_func  # Optional method to run after each forward call of the remote model (simulator)
        super().__init__(*args, **kwargs)

    def close(self):
        if self._model_server is not None:
            self._model_server.close()

    def forward(self):
        if self._model_server is None:
            self._model_server = ModelServer(self._server_address)
            self.name = '{} running on {}'.format(self._model_server.model_name, self._model_server.system_name)

        if self._before_forward_func is not None:
            self._before_forward_func()
        ret = self._model_server.forward()  # Calls the forward run of the remove model (simulator)
        if self._after_forward_func is not None:
            self._after_forward_func()
        return ret


class ConditionalModel(Model):
    def __init__(self, base_model, criterion, criterion_timeout=1e6):
        self._base_model = base_model
        self._criterion = criterion
        self._criterion_timeout = int(criterion_timeout)
        self._traces_total = 1.
        self._traces_accepted = 1.
        # self.name = self._base_model.name
        # self._inference_network = self._base_model._inference_network
        # self._address_dictionary = self._base_model._address_dictionary

    @property
    def acceptance_ratio(self):
        return self._traces_accepted / self._traces_total

    def _trace_generator(self, *args, **kwargs):
        i = 0
        while True:
            i += 1
            if i > self._criterion_timeout:
                raise RuntimeError('ConditionalModel could not sample a trace satisfying the criterion. Timeout ({}) reached.'.format(self._criterion_timeout))
            trace = next(self._base_model._trace_generator(*args, **kwargs))
            self._traces_total += 1.
            if self._criterion(trace):
                self._traces_accepted += 1.
                yield trace
            else:
                continue


class _ParallelModelWorker():
    def __init__(self, model, kwargs):
        self._model = model
        self._kwargs = kwargs

    def run(self, args):
        seed, num_traces, file_name = args[0], args[1], args[2]
        util.seed(seed)
        self._kwargs.update(file_name=file_name)
        self._kwargs.update(num_traces=num_traces)
        self._kwargs.update(silent=True)
        traces = self._model._traces(**self._kwargs)
        log_weights = traces.log_weights_numpy()
        return log_weights


class ParallelModel(Model):
    def __init__(self, base_model, num_workers=None):
        self._base_model = base_model
        if num_workers is None:
            self._num_workers = multiprocessing.cpu_count()
        else:
            self._num_workers = num_workers


    def posterior(self, *args, **kwargs):
        inference_engine = kwargs.get('inference_engine', None)
        print('inference_engine', inference_engine)
        print()
        if inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
            raise ValueError('{} currently not supported by ParallelModel'.format(inference_engine))
        return self._base_model.posterior(*args, **kwargs)

    def _trace_generator(self, *args, **kwargs):
        return self._base_model._trace_generator(*args, **kwargs)

    def _traces(self, num_traces, file_name=None, silent=False, **kwargs):
        if file_name is None:
            file_mode = False
            file_name = util.temp_file_name()
            # /run/user/1000 points to a ram file system in many linux distributions
            # file_name = os.path.join('/run/user/1000', str(uuid.uuid4()))
        else:
            file_mode = True

        num_chunks = self._num_workers
        num_traces_per_chunk = num_traces // num_chunks
        left_over = num_traces - num_traces_per_chunk*num_chunks
        chunks = []
        file_names = []
        seed = util.time_seed()
        for i in range(num_chunks):
            chunk = num_traces_per_chunk
            if i == num_chunks-1 and left_over > 0:
                chunk += left_over
            fn = '{}_chunk_{}_of_{}'.format(file_name, i+1, num_chunks)
            file_names.append(fn)
            chunks.append((seed+i, chunk, fn))

        time_start = time.time()
        if (util._verbosity > 1) and not silent:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1), 'ESS'.ljust(len_str_num_traces+2)))
            prev_duration = 0

        i = -1
        lwi = 0
        log_weights = util.to_tensor(torch.zeros(num_traces))
        pool = multiprocessing.Pool(self._num_workers)
        for j, lw in enumerate(pool.imap(_ParallelModelWorker(self._base_model, kwargs).run, chunks)):
            chunk_len = chunks[j][1]
            i += chunk_len
            lw = torch.from_numpy(lw)[-chunk_len:]
            log_weights[lwi:lwi+chunk_len] = lw
            lwi += chunk_len

            if (util._verbosity > 1) and not silent:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    effective_sample_size = util.effective_sample_size(log_weights[:lwi])
                    if util.has_nan_or_inf(effective_sample_size):
                        effective_sample_size = 0
                    print('{} | {} | {} | {}/{} | {} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, '{:.2f}'.format(effective_sample_size).rjust(len_str_num_traces+2), traces_per_second), end='\r')
                    sys.stdout.flush()

        pool.close()
        pool.join()
        if (util._verbosity > 1) and not silent:
            print()

        if file_mode:
            # Keep files around
            traces = Empirical(concat_empirical_file_names=file_names, file_name=file_name)
        else:
            # Copy everything to memory, delete intermediate files
            traces = Empirical(concat_empirical_file_names=file_names)
            traces.close()
            traces = traces.copy()
            for file_name in file_names:
                os.remove(file_name)

        return traces
