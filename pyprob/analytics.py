from pyprob import PriorInflation


class Analytics():
    def __init__(self, model):
        self._model = model

    def prior_statistics(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, controlled_only=False):
        trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation)
        if controlled_only:
            trace_length_dist = trace_dist.map(lambda trace: len(trace.variables_controlled))
        else:
            trace_length_dist = trace_dist.map(lambda trace: len(trace.variables))
        stats = {}
        stats['trace_length_mean'] = float(trace_length_dist.mean)
        stats['trace_length_stddev'] = float(trace_length_dist.stddev)
        stats['trace_length_min'] = float(trace_length_dist.min)
        stats['trace_length_max'] = float(trace_length_dist.max)
        return stats
