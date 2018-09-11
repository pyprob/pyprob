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

    def prior_statistics(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, file_name=None, *args, **kwargs):
        trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, file_name, bins, log_xscale, log_yscale)

    def posterior_statistics(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe=None, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, file_name=None, *args, **kwargs):
        trace_dist = self._model.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, observe=observe, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, file_name, bins, log_xscale, log_yscale)

    def trace_distribution_statistics(self, trace_dist, use_address_base=True, normalize_per_node=False, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, file_name=None):
        return self._collect_statistics(trace_dist, use_address_base, normalize_per_node, n_most_frequent, file_name, bins, log_xscale, log_yscale)

    def _collect_statistics(self, trace_dist, use_address_base, normalize_per_node, n_most_frequent, file_name, bins, log_xscale, log_yscale):
        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['model_name'] = self._model.name

        master_graph = Graph(trace_dist, use_address_base, n_most_frequent)
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

            file_name_latent_structure_all_pdf = file_name + '_latent_structure_all.pdf'
            print('Rendering latent structure graph (all) to {} ...'.format(file_name_latent_structure_all_pdf))
            master_graph.render_to_file(file_name_latent_structure_all_pdf)

            for i in range(len(trace_stats)):
                trace_id = list(trace_stats.values())[i][1]
                file_name_latent_structure = file_name + '_latent_structure_most_freq_{}_{}.dot'.format(i+1, trace_id)
                print('Saving latent structure graph {} of {} to {} ...'.format(i+1, n_most_frequent, file_name_latent_structure))
                graph = master_graph.get_sub_graph(i)
                graph.render_to_file(file_name_latent_structure, background_graph=master_graph)
        return stats

        # print('Collecting address and trace statistics...')
        # traces = trace_dist.values
        # address_stats = OrderedDict()
        # address_id_to_color = {}
        # address_id_to_variable = {}
        # for trace in traces:
        #     for variable in trace.variables:
        #         address = variable.address_base if use_address_base else variable.address
        #         if address not in address_stats:
        #             address_id = 'A' + str(len(address_stats) + 1)
        #             if variable.control:
        #                 if variable.replace:
        #                     color = '#adff2f'
        #                 else:
        #                     color = '#fa8072'
        #             elif variable.observed:
        #                 color = '#1e90ff'
        #             # elif variable.observable:
        #             #     color = '#1effff'
        #             else:
        #                 color = '#ffd700'
        #             address_stats[address] = [1, address_id, variable]
        #             address_id_to_color[address_id] = color
        #             address_id_to_variable[address_id] = variable
        #         else:
        #             address_stats[address][0] += 1
        #
        # trace_stats = OrderedDict()
        # for trace in traces:
        #     trace_str = ''.join([variable.address_base if use_address_base else variable.address for variable in trace.variables])
        #     if trace_str not in trace_stats:
        #         trace_id = 'T' + str(len(trace_stats) + 1)
        #         address_id_sequence = [address_stats[variable.address_base if use_address_base else variable.address][1] for variable in trace.variables]
        #         trace_stats[trace_str] = [1, trace_id, trace, address_id_sequence]
        #     else:
        #         trace_stats[trace_str][0] += 1
        #
        # stats['addresses'] = len(address_stats)
        # stats['addresses_controlled'] = len([1 for variable in list(address_stats.values()) if variable[2].control])
        # stats['addresses_replaced'] = len([1 for variable in list(address_stats.values()) if variable[2].replace])
        # stats['addresses_observable'] = len([1 for variable in list(address_stats.values()) if variable[2].observable])
        # stats['addresses_observed'] = len([1 for variable in list(address_stats.values()) if variable[2].observed])
        #
        # address_ids = [i for i in range(len(address_stats))]
        # address_weights = []
        # for key, value in address_stats.items():
        #     address_weights.append(value[0])
        # address_id_dist = Empirical(address_ids, weights=address_weights, name='Address ID')
        #
        # trace_ids = [i for i in range(len(trace_stats))]
        # trace_lengths = []
        # trace_lengths_controlled = []
        # trace_execution_times = []
        # trace_weights = []
        # for key, value in trace_stats.items():
        #     trace_lengths.append(len(value[2].variables))
        #     trace_lengths_controlled.append(len(value[2].variables_controlled))
        #     trace_execution_times.append(value[2].execution_time_sec)
        #     trace_weights.append(value[0])
        # trace_id_dist = Empirical(trace_ids, weights=trace_ids, name='Unique trace ID')
        # # trace_id_dist.values = range(len(trace_stats))
        # trace_length_dist = Empirical(trace_lengths, weights=trace_weights, name='Trace length (all)')
        # trace_length_controlled_dist = Empirical(trace_lengths_controlled, weights=trace_weights, name='Trace length (controlled)')
        # trace_execution_time_dist = Empirical(trace_execution_times, weights=trace_weights, name='Trace execution time (s)')
        #
        # stats['traces'] = len(trace_stats)
        # stats['trace_length_min'] = float(trace_length_dist.min)
        # stats['trace_length_max'] = float(trace_length_dist.max)
        # stats['trace_length_mean'] = float(trace_length_dist.mean)
        # stats['trace_length_stddev'] = float(trace_length_dist.stddev)
        # stats['trace_length_controlled_min'] = float(trace_length_controlled_dist.min)
        # stats['trace_length_controlled_max'] = float(trace_length_controlled_dist.max)
        # stats['trace_length_controlled_mean'] = float(trace_length_controlled_dist.mean)
        # stats['trace_length_controlled_stddev'] = float(trace_length_controlled_dist.stddev)
        # stats['trace_execution_time_min'] = float(trace_execution_time_dist.min)
        # stats['trace_execution_time_max'] = float(trace_execution_time_dist.max)
        # stats['trace_execution_time_mean'] = float(trace_execution_time_dist.mean)
        # stats['trace_execution_time_stddev'] = float(trace_execution_time_dist.stddev)
        #
        # print('Generating latent structure master graph...')
        # sorted_trace_stats = OrderedDict(sorted(dict(trace_stats).items(), key=lambda x: x[1][0], reverse=True))
        # if n_most_frequent is None:
        #     n_most_frequent = len(trace_stats)
        # sorted_trace_stats = dict(islice(sorted_trace_stats.items(), n_most_frequent))
        #
        # master_graph_nodes = {}
        # master_graph_edges = {}
        # for key, value in sorted_trace_stats.items():
        #     count = value[0]
        #     address_id_sequence = value[3]
        #     for address_id in address_id_sequence:
        #         if address_id in master_graph_nodes:
        #             master_graph_nodes[address_id] += count
        #         else:
        #             master_graph_nodes[address_id] = count
        #     for left, right in zip(address_id_sequence, address_id_sequence[1:]):
        #         if (left, right) in master_graph_edges:
        #             master_graph_edges[(left, right)] += count
        #         else:
        #             master_graph_edges[(left, right)] = count
        #
        # # networkx code for later
        # # nx_master_graph = nx.DiGraph()
        # # nx_master_graph.add_weighted_edges_from([(key[0], key[1], value) for key, value in master_graph_edges.items()])
        # # nx_master_graph.add_nodes_from([(key, {'variable': address_id_to_variable[key], 'weight': value}) for key, value in master_graph_nodes.items()])
        #
        # print('Generating latent structure master graph...')
        # sorted_trace_stats = OrderedDict(sorted(dict(trace_stats).items(), key=lambda x: x[1][0], reverse=True))
        # if n_most_frequent is None:
        #     n_most_frequent = len(trace_stats)
        # sorted_trace_stats = dict(islice(sorted_trace_stats.items(), n_most_frequent))
        #
        # master_graph_nodes = {}
        # master_graph_edges = {}
        # for key, value in sorted_trace_stats.items():
        #     count = value[0]
        #     address_id_sequence = value[3]
        #     for address_id in address_id_sequence:
        #         if address_id in master_graph_nodes:
        #             master_graph_nodes[address_id] += count
        #         else:
        #             master_graph_nodes[address_id] = count
        #     for left, right in zip(address_id_sequence, address_id_sequence[1:]):
        #         if (left, right) in master_graph_edges:
        #             master_graph_edges[(left, right)] += count
        #         else:
        #             master_graph_edges[(left, right)] = count
        #
        # master_graph = pydotplus.graphviz.Dot(graph_type='digraph', rankdir='LR')
        # transition_count_max = 0
        # for edge, count in master_graph_edges.items():
        #     if count > transition_count_max:
        #         transition_count_max = count
        #     nodes = master_graph.get_node(edge[0])
        #     if len(nodes) > 0:
        #         n0 = nodes[0]
        #     else:
        #         n0 = pydotplus.Node(edge[0])
        #         master_graph.add_node(n0)
        #     nodes = master_graph.get_node(edge[1])
        #     if len(nodes) > 0:
        #         n1 = nodes[0]
        #     else:
        #         n1 = pydotplus.Node(edge[1])
        #         master_graph.add_node(n1)
        #     master_graph.add_edge(pydotplus.Edge(n0, n1, weight=count))
        #     for node in master_graph.get_nodes():
        #         node.set_color('#cccccc')
        #         node.set_fontcolor('#cccccc')
        #     for edge in master_graph.get_edges():
        #         edge.set_color('#cccccc')
        #
        # master_graph_annotated = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())
        # for node in master_graph_annotated.get_nodes():
        #     # color = util.rgb_to_hex(util.rgb_blend((1, 1, 1), (1, 0, 0), address_id_to_count[node.obj_dict['name']] / address_id_count_total))
        #     address_id = node.obj_dict['name']
        #     node.set_style('filled')
        #     node.set_fillcolor(address_id_to_color[address_id])
        #     node.set_color('black')
        #     node.set_fontcolor('black')
        #     count = master_graph_nodes[address_id]
        #     color_factor = 0.75 * (math.exp(1. - count / transition_count_max) - 1.) / (math.e - 1.)
        #     # color = rgb_to_hex((color_factor, color_factor, color_factor))
        #     # node.set_color(color)
        #     node.set_penwidth(max(0.1, 4 * (1 - color_factor)))
        # for edge in master_graph_annotated.get_edges():
        #     (left, right) = edge.obj_dict['points']
        #     count = master_graph_edges[(left, right)]
        #     edge.set_label('\"{:,.6f}\"'.format(count / transition_count_max))
        #     # color_factor = 1 - max(0.15, min(1, count/transition_count_max))
        #     color_factor = 0.75 * (math.exp(1. - count / transition_count_max) - 1.) / (math.e - 1.)
        #     color = util.rgb_to_hex((color_factor, color_factor, color_factor))
        #     # edge.set_color('black')
        #     edge.set_color(color)
        #     # edge.set_penwidth(max(0.1, 5 * count / transition_count_max))
        #
        # graphs = []
        # i = 0
        # for key, value in sorted_trace_stats.items():
        #     i += 1
        #     count = value[0]
        #     address_id_sequence = value[3]
        #     print('Generating latent structure trace graph {} of {}...'.format(i, n_most_frequent))
        #
        #     graph_nodes = {}
        #     graph_edges = {}
        #     for address_id in address_id_sequence:
        #         if address_id in graph_nodes:
        #             graph_nodes[address_id] += count
        #         else:
        #             graph_nodes[address_id] = count
        #     for left, right in zip(address_id_sequence, address_id_sequence[1:]):
        #         if (left, right) in graph_edges:
        #             graph_edges[(left, right)] += count
        #         else:
        #             graph_edges[(left, right)] = count
        #
        #     graph = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())
        #     transition_count_max = 0
        #     for edge, count in graph_edges.items():
        #         if count > transition_count_max:
        #             transition_count_max = count
        #
        #     for node in graph.get_nodes():
        #         # color = util.rgb_to_hex(util.rgb_blend((1, 1, 1), (1, 0, 0), address_id_to_count[node.obj_dict['name']] / address_id_count_total))
        #         address_id = node.obj_dict['name']
        #         if address_id in graph_nodes:
        #             count = graph_nodes[address_id]
        #             node.set_style('filled')
        #             node.set_fillcolor(address_id_to_color[address_id])
        #             node.set_color('black')
        #             node.set_fontcolor('black')
        #             color_factor = 0.75 * (math.exp(1. - count / transition_count_max) - 1.) / (math.e - 1.)
        #             # color = rgb_to_hex((color_factor, color_factor, color_factor))
        #             # node.set_color(color)
        #             node.set_penwidth(max(0.1, 4 * (1 - color_factor)))
        #     for edge in graph.get_edges():
        #         (left, right) = edge.obj_dict['points']
        #         if (left, right) in graph_edges:
        #             count = graph_edges[(left, right)]
        #             edge.set_label('\"{:,.6f}\"'.format(count / transition_count_max))
        #             # color_factor = 1 - max(0.15, min(1, count/transition_count_max))
        #             color_factor = 0.75 * (math.exp(1. - count / transition_count_max) - 1.) / (math.e - 1.)
        #             color = util.rgb_to_hex((color_factor, color_factor, color_factor))
        #             # edge.set_color('black')
        #             edge.set_color(color)
        #             # edge.set_penwidth(max(0.1, 5 * count / transition_count_max))
        #
        #     graphs.append(graph)
        #
        # if file_name is not None:
        #     file_name_stats = file_name + '.txt'
        #     print('Saving analytics information to {} ...'.format(file_name_stats))
        #     with open(file_name_stats, 'w') as file:
        #         file.write('pyprob analytics report\n')
        #         for key, value in stats.items():
        #             file.write('{}: {}\n'.format(key, value))
        #
        #     file_name_addresses = file_name + '_addresses.csv'
        #     print('Saving addresses to {} ...'.format(file_name_addresses))
        #     with open(file_name_addresses, 'w') as file:
        #         file.write('address_id, frequency, name, controlled, replaced, observable, observed, {}\n'.format('address_base' if use_address_base else 'address'))
        #         for key, value in address_stats.items():
        #             name = '' if value[2].name is None else value[2].name
        #             file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value[1], value[0], name, value[2].control, value[2].replace, value[2].observable, value[2].observed, key))
        #
        #     file_name_traces = file_name + '_traces.csv'
        #     print('Saving addresses to {} ...'.format(file_name_traces))
        #     with open(file_name_traces, 'w') as file:
        #         file.write('trace_id, frequency, length, length_controlled, address_id_sequence\n')
        #         for key, value in trace_stats.items():
        #             file.write('{}, {}, {}, {}, {}\n'.format(value[1], value[0], len(value[2].variables), len(value[2].variables_controlled), ' '.join(value[3])))
        #
        #     file_name_address_id_dist = file_name + '_address_ids.pdf'
        #     print('Saving trace type distribution to {} ...'.format(file_name_address_id_dist))
        #     address_id_dist.plot_histogram(bins=range(len(address_stats)), xticks=range(len(address_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_address_id_dist)
        #
        #     file_name_trace_id_dist = file_name + '_trace_ids.pdf'
        #     print('Saving trace type distribution to {} ...'.format(file_name_trace_id_dist))
        #     trace_id_dist.plot_histogram(bins=range(len(trace_stats)), xticks=range(len(trace_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_id_dist)
        #
        #     file_name_trace_length_dist = file_name + '_trace_length_all.pdf'
        #     print('Saving trace length (all) distribution to {} ...'.format(file_name_trace_length_dist))
        #     trace_length_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_dist)
        #
        #     file_name_trace_length_controlled_dist = file_name + '_trace_length_controlled.pdf'
        #     print('Saving trace length (controlled) distribution to {} ...'.format(file_name_trace_length_controlled_dist))
        #     trace_length_controlled_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_controlled_dist)
        #
        #     file_name_trace_execution_time_dist = file_name + '_trace_execution_time.pdf'
        #     print('Saving trace execution time distribution to {} ...'.format(file_name_trace_execution_time_dist))
        #     trace_execution_time_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_execution_time_dist)
        #
        #     file_name_latent_structure_all = file_name + '_latent_structure_all.dot'
        #     print('Saving latent structure graph (all) to {} ...'.format(file_name_latent_structure_all))
        #     with open(file_name_latent_structure_all, 'w') as file:
        #         file.write(master_graph_annotated.to_string())
        #
        #     file_name_latent_structure_all_pdf = file_name + '_latent_structure_all.pdf'
        #     print('Rendering latent structure graph (all) to {} ...'.format(file_name_latent_structure_all_pdf))
        #     status, result = subprocess.getstatusoutput('dot -Tpdf {} -o {}'.format(file_name_latent_structure_all, file_name_latent_structure_all_pdf))
        #     if status != 0:
        #         print('Could not render file {}. Check that GraphViz is installed.'.format(file_name_latent_structure_all))
        #
        #     for i in range(n_most_frequent):
        #         trace_id = list(sorted_trace_stats.values())[i][1]
        #         file_name_latent_structure = file_name + '_latent_structure_most_freq_{}_{}.dot'.format(i+1, trace_id)
        #         print('Saving latent structure graph {} of {} to {} ...'.format(i+1, n_most_frequent, file_name_latent_structure))
        #         with open(file_name_latent_structure, 'w') as file:
        #             file.write(graphs[i].to_string())
        #
        #         file_name_latent_structure_pdf = file_name + '_latent_structure_most_freq_{}_{}.pdf'.format(i+1, trace_id)
        #         print('Rendering latent structure graph {} of {} to {} ...'.format(i+1, n_most_frequent, file_name_latent_structure_pdf))
        #         status, result = subprocess.getstatusoutput('dot -Tpdf {} -o {}'.format(file_name_latent_structure, file_name_latent_structure_pdf))
        #         if status != 0:
        #             print('Could not render file {}. Check that GraphViz is installed.'.format(file_name_latent_structure))

        # return stats
