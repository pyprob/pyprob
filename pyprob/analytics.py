import torch
import os
from collections import OrderedDict

from . import __version__, util, PriorInflation, InferenceEngine
from .distributions import Empirical
from .graph import Graph


class Analytics():
    def __init__(self, model):
        self._model = model

    def prior_graph(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, use_address_base=True, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, report_dir=None, trace_dist=None, *args, **kwargs):
        if trace_dist is None:
            trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, n_most_frequent, base_graph, report_dir, None, 'prior', bins, log_xscale, log_yscale)

    def posterior_graph(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe=None, use_address_base=True, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, report_dir=None, trace_dist=None, *args, **kwargs):
        if trace_dist is None:
            trace_dist = self._model.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, observe=observe, *args, **kwargs)
        return self._collect_statistics(trace_dist, use_address_base, n_most_frequent, base_graph, report_dir, inference_engine.name, 'posterior', bins, log_xscale, log_yscale)

    def _collect_statistics(self, trace_dist, use_address_base, n_most_frequent, base_graph, report_dir, report_sub_dir, report_name, bins, log_xscale, log_yscale):
        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['model_name'] = self._model.name

        print('Building graph...')
        if base_graph is not None:
            use_address_base = base_graph.use_address_base
        master_graph = Graph(trace_dist=trace_dist, use_address_base=use_address_base, n_most_frequent=n_most_frequent, base_graph=base_graph)
        address_stats = master_graph.address_stats
        stats['addresses'] = len(address_stats)
        stats['addresses_controlled'] = len([1 for value in list(address_stats.values()) if value['variable'].control])
        stats['addresses_replaced'] = len([1 for value in list(address_stats.values()) if value['variable'].replace])
        stats['addresses_observable'] = len([1 for value in list(address_stats.values()) if value['variable'].observable])
        stats['addresses_observed'] = len([1 for value in list(address_stats.values()) if value['variable'].observed])

        address_ids = [i for i in range(len(address_stats))]
        address_weights = []
        for key, value in address_stats.items():
            address_weights.append(value['count'])
        address_id_dist = Empirical(address_ids, weights=address_weights, name='Address ID')

        trace_stats = master_graph.trace_stats
        trace_ids = [i for i in range(len(trace_stats))]
        trace_weights = []
        for key, value in trace_stats.items():
            trace_weights.append(value['count'])
        trace_id_dist = Empirical(trace_ids, weights=trace_ids, name='Unique trace ID')
        trace_dist_unweighted = trace_dist.unweighted()
        trace_length_dist = trace_dist_unweighted.map(lambda trace: trace.length).rename('Trace length (all)')
        trace_length_controlled_dist = trace_dist_unweighted.map(lambda trace: trace.length_controlled).rename('Trace length (controlled)')
        trace_execution_time_dist = trace_dist_unweighted.map(lambda trace: trace.execution_time_sec).rename('Trace execution time (s)')

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

        if report_dir is not None:
            report_root = os.path.join(report_dir, report_name)
            if report_sub_dir is not None:
                report_root = os.path.join(report_root, report_sub_dir)
            if not os.path.exists(report_root):
                print('Directory does not exist, creating: {}'.format(report_root))
                os.makedirs(report_root)
            file_name_stats = os.path.join(report_root, report_name + '_stats.txt')
            print('Saving analytics information to {} ...'.format(file_name_stats))
            with open(file_name_stats, 'w') as file:
                file.write('pyprob analytics report\n')
                for key, value in stats.items():
                    file.write('{}: {}\n'.format(key, value))

            file_name_addresses = os.path.join(report_root, report_name + '_addresses.csv')
            print('Saving addresses to {} ...'.format(file_name_addresses))
            with open(file_name_addresses, 'w') as file:
                file.write('address_id, weight, name, controlled, replaced, observable, observed, {}\n'.format('address_base' if use_address_base else 'address'))
                for key, value in address_stats.items():
                    name = '' if value['variable'].name is None else value['variable'].name
                    file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value['address_id'], value['weight'], name, value['variable'].control, value['variable'].replace, value['variable'].observable, value['variable'].observed, key))

            file_name_traces = os.path.join(report_root, report_name + '_traces.csv')
            print('Saving addresses to {} ...'.format(file_name_traces))
            with open(file_name_traces, 'w') as file:
                file.write('trace_id, weight, length, length_controlled, address_id_sequence\n')
                for key, value in trace_stats.items():
                    file.write('{}, {}, {}, {}, {}\n'.format(value['trace_id'], value['weight'], len(value['trace'].variables), len(value['trace'].variables_controlled), ' '.join(value['address_id_sequence'])))

            file_name_address_id_dist = os.path.join(report_root, report_name + '_address_ids.pdf')
            print('Saving trace type distribution to {} ...'.format(file_name_address_id_dist))
            address_id_dist.plot_histogram(bins=range(len(address_stats)), xticks=range(len(address_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_address_id_dist)

            file_name_trace_id_dist = os.path.join(report_root, report_name + '_trace_ids.pdf')
            print('Saving trace type distribution to {} ...'.format(file_name_trace_id_dist))
            trace_id_dist.plot_histogram(bins=range(len(trace_stats)), xticks=range(len(trace_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_id_dist)

            file_name_trace_length_dist = os.path.join(report_root, report_name + '_trace_length_all.pdf')
            print('Saving trace length (all) distribution to {} ...'.format(file_name_trace_length_dist))
            trace_length_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_dist)

            file_name_trace_length_controlled_dist = os.path.join(report_root, report_name + '_trace_length_controlled.pdf')
            print('Saving trace length (controlled) distribution to {} ...'.format(file_name_trace_length_controlled_dist))
            trace_length_controlled_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_controlled_dist)

            file_name_trace_execution_time_dist = os.path.join(report_root, report_name + '_trace_execution_time.pdf')
            print('Saving trace execution time distribution to {} ...'.format(file_name_trace_execution_time_dist))
            trace_execution_time_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_execution_time_dist)

            report_latent_root = os.path.join(report_root, 'latent_structure')
            if not os.path.exists(report_latent_root):
                print('Directory does not exist, creating: {}'.format(report_latent_root))
                os.makedirs(report_latent_root)
            file_name_latent_structure_all_pdf = os.path.join(report_latent_root, report_name + '_latent_structure_all')
            print('Rendering latent structure graph (all) to {} ...'.format(file_name_latent_structure_all_pdf))
            if base_graph is None:
                master_graph.render_to_file(file_name_latent_structure_all_pdf)
            else:
                master_graph.render_to_file(file_name_latent_structure_all_pdf, background_graph=base_graph)

            for i in range(len(trace_stats)):
                trace_id = list(trace_stats.values())[i]['trace_id']
                file_name_latent_structure = os.path.join(report_latent_root, report_name + '_latent_structure_most_freq_{}_{}'.format(i+1, trace_id))
                print('Saving latent structure graph {} of {} to {} ...'.format(i+1, len(trace_stats), file_name_latent_structure))
                graph = master_graph.get_sub_graph(i)
                if base_graph is None:
                    graph.render_to_file(file_name_latent_structure, background_graph=master_graph)
                else:
                    graph.render_to_file(file_name_latent_structure, background_graph=base_graph)

            print('Rendering distributions...')
            report_distribution_root = os.path.join(report_root, 'distributions')
            if not os.path.exists(report_distribution_root):
                print('Directory does not exist, creating: {}'.format(report_distribution_root))
                os.makedirs(report_distribution_root)

            for key, value in address_stats.items():
                address_id = value['address_id']
                variable = value['variable']
                can_render = True
                try:
                    if use_address_base:
                        address_base = variable.address_base
                        dist = trace_dist.filter(lambda trace: address_base in trace.variables_dict_address_base).map(lambda trace: util.to_tensor(trace.variables_dict_address_base[address_base].value)).filter(lambda v: torch.is_tensor(v)).filter(lambda v: v.nelement() == 1)
                    else:
                        address = variable.address
                        dist = trace_dist.filter(lambda trace: address in trace.variables_dict_address).map(lambda trace: util.to_tensor(trace.variables_dict_address[address].value)).filter(lambda v: torch.is_tensor(v)).filter(lambda v: v.nelement() == 1)

                    dist.rename(address_id + '' if variable.name is None else '{} (name: {})'.format(address_id, variable.name))
                    if dist.length == 0:
                        can_render = False
                except:
                    can_render = False

                if can_render:
                    file_name_dist = os.path.join(report_distribution_root, '{}_{}_distribution.pdf'.format(address_id, report_name))
                    print('Saving distribution to {} ...'.format(file_name_dist))
                    dist.plot_histogram(bins=bins, color='black', show=False, file_name=file_name_dist)
                    if report_sub_dir is not None:
                        if report_sub_dir == 'IMPORTANCE_SAMPLING':
                            file_name_dist = os.path.join(report_distribution_root, '{}_{}_distribution.pdf'.format(address_id, 'proposal'))
                            print('Saving distribution to {} ...'.format(file_name_dist))
                            dist.unweighted().plot_histogram(bins=bins, color='black', show=False, file_name=file_name_dist)

                else:
                    print('Cannot render histogram for {} because it is not scalar valued. Example value: {}'.format(address_id, variable.value))

        return master_graph, stats
