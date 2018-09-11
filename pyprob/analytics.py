import torch
import networkx as nx
import math
from collections import OrderedDict
from itertools import islice
import pydotplus
import subprocess

from . import __version__, util, PriorInflation, InferenceEngine
from .distributions import Empirical
from .graph import Graph


class Analytics():
    def __init__(self, model):
        self._model = model

    def prior_graph(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, file_name=None, *args, **kwargs):
        trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, base_graph, file_name, bins, log_xscale, log_yscale)

    def posterior_graph(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe=None, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, file_name=None, *args, **kwargs):
        trace_dist = self._model.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, observe=observe, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, base_graph, file_name, bins, log_xscale, log_yscale)

    def trace_distribution_graph(self, trace_dist, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, file_name=None):
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, base_graph, file_name, bins, log_xscale, log_yscale)

    def _collect_statistics(self, trace_dist, use_address_base, normalize_per_node, n_most_frequent, base_graph, file_name, bins, log_xscale, log_yscale):
        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['model_name'] = self._model.name

        if base_graph is not None:
            use_address_base = base_graph.use_address_base
        master_graph = Graph(trace_dist=trace_dist, use_address_base=use_address_base, n_most_frequent=n_most_frequent, base_graph=base_graph)
        address_stats = master_graph.address_stats
        stats['addresses'] = len(address_stats)
        stats['addresses_controlled'] = len([1 for variable in list(address_stats.values()) if variable[2].control])
        stats['addresses_replaced'] = len([1 for variable in list(address_stats.values()) if variable[2].replace])
        stats['addresses_observable'] = len([1 for variable in list(address_stats.values()) if variable[2].observable])
        stats['addresses_observed'] = len([1 for variable in list(address_stats.values()) if variable[2].observed])

        address_ids = [i for i in range(len(address_stats))]
        address_weights = []
        for key, value in address_stats.items():
            address_weights.append(value[0])
        address_id_dist = Empirical(address_ids, weights=address_weights, name='Address ID')

        trace_stats = master_graph.trace_stats
        trace_ids = [i for i in range(len(trace_stats))]
        trace_lengths = []
        trace_lengths_controlled = []
        trace_execution_times = []
        trace_weights = []
        for key, value in trace_stats.items():
            trace_lengths.append(len(value[2].variables))
            trace_lengths_controlled.append(len(value[2].variables_controlled))
            trace_execution_times.append(value[2].execution_time_sec)
            trace_weights.append(value[0])
        trace_id_dist = Empirical(trace_ids, weights=trace_ids, name='Unique trace ID')
        # trace_id_dist.values = range(len(trace_stats))
        trace_length_dist = Empirical(trace_lengths, weights=trace_weights, name='Trace length (all)')
        trace_length_controlled_dist = Empirical(trace_lengths_controlled, weights=trace_weights, name='Trace length (controlled)')
        trace_execution_time_dist = Empirical(trace_execution_times, weights=trace_weights, name='Trace execution time (s)')

        stats['traces'] = len(trace_stats)
        stats['trace_length_min'] = float(trace_length_dist.min)
        stats['trace_length_max'] = float(trace_length_dist.max)
        stats['trace_length_mean'] = float(trace_length_dist.mean)
        stats['trace_length_stddev'] = float(trace_length_dist.stddev)
        stats['trace_length_controlled_min'] = float(trace_length_controlled_dist.min)
        stats['trace_length_controlled_max'] = float(trace_length_controlled_dist.max)
        stats['trace_length_controlled_mean'] = float(trace_length_controlled_dist.mean)
        stats['trace_length_controlled_stddev'] = float(trace_length_controlled_dist.stddev)
        stats['trace_execution_time_min'] = float(trace_execution_time_dist.min)
        stats['trace_execution_time_max'] = float(trace_execution_time_dist.max)
        stats['trace_execution_time_mean'] = float(trace_execution_time_dist.mean)
        stats['trace_execution_time_stddev'] = float(trace_execution_time_dist.stddev)

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
                file.write('address_id, frequency, name, controlled, replaced, observable, observed, {}\n'.format('address_base' if use_address_base else 'address'))
                for key, value in address_stats.items():
                    name = '' if value[2].name is None else value[2].name
                    file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value[1], value[0], name, value[2].control, value[2].replace, value[2].observable, value[2].observed, key))

            file_name_traces = file_name + '_traces.csv'
            print('Saving addresses to {} ...'.format(file_name_traces))
            with open(file_name_traces, 'w') as file:
                file.write('trace_id, frequency, length, length_controlled, address_id_sequence\n')
                for key, value in trace_stats.items():
                    file.write('{}, {}, {}, {}, {}\n'.format(value[1], value[0], len(value[2].variables), len(value[2].variables_controlled), ' '.join(value[3])))

            file_name_address_id_dist = file_name + '_address_ids.pdf'
            print('Saving trace type distribution to {} ...'.format(file_name_address_id_dist))
            address_id_dist.plot_histogram(bins=range(len(address_stats)), xticks=range(len(address_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_address_id_dist)

            file_name_trace_id_dist = file_name + '_trace_ids.pdf'
            print('Saving trace type distribution to {} ...'.format(file_name_trace_id_dist))
            trace_id_dist.plot_histogram(bins=range(len(trace_stats)), xticks=range(len(trace_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_id_dist)

            file_name_trace_length_dist = file_name + '_trace_length_all.pdf'
            print('Saving trace length (all) distribution to {} ...'.format(file_name_trace_length_dist))
            trace_length_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_dist)

            file_name_trace_length_controlled_dist = file_name + '_trace_length_controlled.pdf'
            print('Saving trace length (controlled) distribution to {} ...'.format(file_name_trace_length_controlled_dist))
            trace_length_controlled_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_controlled_dist)

            file_name_trace_execution_time_dist = file_name + '_trace_execution_time.pdf'
            print('Saving trace execution time distribution to {} ...'.format(file_name_trace_execution_time_dist))
            trace_execution_time_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_execution_time_dist)

            file_name_latent_structure_all_pdf = file_name + '_latent_structure_all'
            print('Rendering latent structure graph (all) to {} ...'.format(file_name_latent_structure_all_pdf))
            if base_graph is None:
                master_graph.render_to_file(file_name_latent_structure_all_pdf)
            else:
                master_graph.render_to_file(file_name_latent_structure_all_pdf, background_graph=base_graph)

            for i in range(len(trace_stats)):
                trace_id = list(trace_stats.values())[i][1]
                file_name_latent_structure = file_name + '_latent_structure_most_freq_{}_{}'.format(i+1, trace_id)
                print('Saving latent structure graph {} of {} to {} ...'.format(i+1, len(trace_stats), file_name_latent_structure))
                graph = master_graph.get_sub_graph(i)
                if base_graph is None:
                    graph.render_to_file(file_name_latent_structure, background_graph=master_graph)
                else:
                    graph.render_to_file(file_name_latent_structure, background_graph=base_graph)

        return master_graph, stats
