import torch.nn as nn
import time
import sys

from .distributions import Empirical
from . import util, state, TraceMode, PriorInflation, InferenceEngine


class Model(nn.Module):
    def __init__(self, name='Unnamed pyprob model'):
        super().__init__()
        self.name = name
        self._inference_network = None

    def forward(self):
        raise NotImplementedError()

    def _trace_generator(self, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, metropolis_hastings_trace=None, *args, **kwargs):
        while True:
            state.begin_trace(self.forward, trace_mode, prior_inflation, inference_engine, inference_network, observe, metropolis_hastings_trace)
            result = self.forward(*args, **kwargs)
            trace = state.end_trace(result)
            yield trace

    def _traces(self, num_traces=10, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, map_func=None, observe=None, *args, **kwargs):
        generator = self._trace_generator(trace_mode=trace_mode, prior_inflation=prior_inflation, inference_engine=inference_engine, inference_network=inference_network, observe=observe, *args, **kwargs)
        traces = []
        log_weights = []
        time_start = time.time()
        if util._verbosity > 1:
            len_str_num_traces = len(str(num_traces))
            print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
            prev_duration = 0
        for i in range(num_traces):
            if util._verbosity > 1:
                duration = time.time() - time_start
                if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                    prev_duration = duration
                    traces_per_second = (i + 1) / duration
                    print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                    sys.stdout.flush()
            trace = next(generator)
            if map_func is not None:
                traces.append(map_func(trace))
            else:
                traces.append(trace)
            log_weights.append(trace.log_importance_weight)
        if util._verbosity > 1:
            print()
        return traces, log_weights

    def prior_traces(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=None, *args, **kwargs):
        traces, _ = self._traces(num_traces=num_traces, trace_mode=TraceMode.PRIOR, prior_inflation=prior_inflation, map_func=map_func, *args, **kwargs)
        return Empirical(traces, name='Prior -- {}'.format(self.name))

    def prior_distribution(self, num_traces=10, prior_inflation=PriorInflation.DISABLED, map_func=lambda trace: trace.result, *args, **kwargs):
        return self.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, map_func=map_func, *args, **kwargs)

    def posterior_traces(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, burn_in=None, initial_trace=None, map_func=None, observe=None, *args, **kwargs):
        if (inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK) and (self._inference_network is None):
            raise RuntimeError('Cannot run inference engine IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK because no inference network for this model is available. Use learn_inference_network or load_inference_network first.')
        if burn_in is not None:
            if burn_in >= num_traces:
                raise ValueError('burn_in must be less than num_traces')
        else:
            # Default burn_in
            burn_in = int(min(num_traces / 10, 1000))

        if inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
            traces, log_weights = self._traces(num_traces=num_traces, trace_mode=TraceMode.POSTERIOR, inference_engine=inference_engine, inference_network=None, map_func=map_func, observe=observe, *args, **kwargs)
            name = 'Posterior, importance sampling (with proposal = prior), num_traces={:,}'.format(num_traces)
        else:
            raise NotImplementedError()

        return Empirical(traces, log_weights, name=name)

    def posterior_distribution(self, num_traces=10, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, burn_in=None, initial_trace=None, map_func=lambda trace: trace.result, observe=None, *args, **kwargs):
        return self.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, burn_in=burn_in, initial_trace=initial_trace, map_func=map_func, observe=observe, *args, **kwargs)
