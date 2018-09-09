import torch
from collections import OrderedDict

from pyprob import __version__, PriorInflation, InferenceEngine


class Analytics():
    def __init__(self, model):
        self._model = model

    def prior_statistics(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, controlled_only=False, file_name=None, *args, **kwargs):
        trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, *args, **kwargs)
        return self._collect_statistics(trace_dist, controlled_only, file_name)

    def posterior_statistics(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe=None, controlled_only=False, file_name=None, *args, **kwargs):
        trace_dist = self._model.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, observe=observe, *args, **kwargs)
        return self._collect_statistics(trace_dist, controlled_only, file_name)

    def _collect_statistics(self, trace_dist, controlled_only, file_name):
        if controlled_only:
            trace_length_dist = trace_dist.map(lambda trace: len(trace.variables_controlled))
        else:
            trace_length_dist = trace_dist.map(lambda trace: len(trace.variables))
        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['model_name'] = self._model.name
        stats['trace_length_mean'] = float(trace_length_dist.mean)
        stats['trace_length_stddev'] = float(trace_length_dist.stddev)
        stats['trace_length_min'] = float(trace_length_dist.min)
        stats['trace_length_max'] = float(trace_length_dist.max)

        print('Collecting address and trace statistics...')
        traces = trace_dist.values
        address_stats = OrderedDict()
        trace_stats = OrderedDict()
        for trace in traces:
            for variable in trace.variables:
                address = variable.address_base
                if address not in address_stats:
                    address_id = 'A' + str(len(address_stats) + 1)
                    address_stats[address] = [1, address_id, variable]
                else:
                    address_stats[address][0] += 1

        for trace in traces:
            trace_str = ''.join([variable.address_base for variable in trace.variables])
            if trace_str not in trace_stats:
                trace_id = 'T' + str(len(trace_stats) + 1)
                address_id_sequence = [address_stats[variable.address_base][1] for variable in trace.variables]
                trace_stats[trace_str] = [1, trace_id, trace, address_id_sequence]
            else:
                trace_stats[trace_str][0] += 1

        stats['num_addresses'] = len(address_stats)
        stats['num_addresses_controlled'] = len([1 for variable in list(address_stats.values()) if variable[2].control])
        stats['num_addresses_replaced'] = len([1 for variable in list(address_stats.values()) if variable[2].replace])
        stats['num_addresses_observable'] = len([1 for variable in list(address_stats.values()) if variable[2].observable])
        stats['num_addresses_observed'] = len([1 for variable in list(address_stats.values()) if variable[2].observed])
        stats['num_traces'] = len(trace_stats)

        if file_name is not None:
            file_name_stats = file_name + '.txt'
            print('Saving analytics information to {} ...'.format(file_name_stats))
            with open(file_name_stats, 'w') as file:
                file.write('pyprob analytics report\n')
                for key, value in stats.items():
                    file.write('{}: {}\n'.format(key, value))

            file_name_addresses = file_name + '_addresses.csv'
            print('Saving addresses to {} ...'.format(file_name_addresses))
            with open(file_name_addresses, 'w') as file:
                file.write('address_id, frequency, name, controlled, replaced, observable, observed, address_base\n')
                for key, value in address_stats.items():
                    name = '' if value[2].name is None else value[2].name
                    file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value[1], value[0], name, value[2].control, value[2].replace, value[2].observable, value[2].observed, key))

            file_name_traces = file_name + '_traces.csv'
            print('Saving addresses to {} ...'.format(file_name_traces))
            with open(file_name_traces, 'w') as file:
                file.write('trace_id, frequency, length, length_controlled, address_id_sequence\n')
                for key, value in trace_stats.items():
                    file.write('{}, {}, {}, {}, {}\n'.format(value[1], value[0], len(value[2].variables), len(value[2].variables_controlled), ' '.join(value[3])))
        return stats
