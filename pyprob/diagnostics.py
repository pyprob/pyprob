import torch
import os
from collections import OrderedDict, defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
import csv
from torch.distributions.kl import kl_divergence

from . import __version__, util
from .distributions import Empirical
from .graph import Graph
from .trace import Trace


def _address_stats(trace_dist, use_address_base=True, reuse_ids_from_address_stats=None):
    addresses = {}
    address_id_to_variable = {}
    if reuse_ids_from_address_stats is not None:
        address_ids = reuse_ids_from_address_stats['address_ids']
        address_base_ids = reuse_ids_from_address_stats['address_base_ids']
    else:
        address_ids = {}
        address_base_ids = {}
    for i in range(trace_dist.length):
        trace = trace_dist._get_value(i)
        trace_weight = float(trace_dist._get_weight(i))
        for variable in trace.variables:
            address_base = variable.address_base
            address = variable.address
            key = address_base if use_address_base else address
            if key in addresses:
                addresses[key]['count'] += 1
                addresses[key]['weight'] += trace_weight
            else:
                if key in address_ids:
                    address_id = address_ids[key]
                else:
                    if use_address_base:
                        if address_base.startswith('__A'):
                            address_id = address_base[2:]
                        else:
                            address_id = 'A' + str(len(address_ids) + 1)
                    else:
                        if address_base.startswith('__A'):
                            address_id = address[2:]
                        else:
                            if address_base not in address_base_ids:
                                address_base_id = 'A' + str(len(address_base_ids) + 1)
                                address_base_ids[address_base] = address_base_id
                            address_id = address_base_ids[address_base] + '__' + str(variable.instance)
                    address_ids[key] = address_id
                addresses[key] = {'count': 1, 'weight': trace_weight, 'address_id': address_id, 'variable': variable}
                address_id_to_variable[address_id] = variable
    addresses = OrderedDict(sorted(addresses.items(), key=lambda v: util.address_id_to_int(v[1]['address_id'])))
    addresses_extra = OrderedDict()
    addresses_extra['pyprob_version'] = __version__
    addresses_extra['torch_version'] = torch.__version__
    addresses_extra['num_distribution_elements'] = len(trace_dist)
    addresses_extra['addresses'] = len(addresses)
    addresses_extra['addresses_controlled'] = len([1 for value in list(addresses.values()) if value['variable'].control])
    addresses_extra['addresses_replaced'] = len([1 for value in list(addresses.values()) if value['variable'].replace])
    addresses_extra['addresses_observable'] = len([1 for value in list(addresses.values()) if value['variable'].observable])
    addresses_extra['addresses_observed'] = len([1 for value in list(addresses.values()) if value['variable'].observed])
    addresses_extra['addresses_tagged'] = len([1 for value in list(addresses.values()) if value['variable'].tagged])
    return {'addresses': addresses, 'addresses_extra': addresses_extra, 'address_base_ids': address_base_ids, 'address_ids': address_ids, 'address_id_to_variable': address_id_to_variable}


def _trace_stats(trace_dist, use_address_base=True, reuse_ids_from_address_stats=None, reuse_ids_from_trace_stats=None):
    address_stats = _address_stats(trace_dist, use_address_base=use_address_base, reuse_ids_from_address_stats=reuse_ids_from_address_stats)
    addresses = address_stats['addresses']
    traces = {}
    if reuse_ids_from_trace_stats is not None:
        trace_ids = reuse_ids_from_trace_stats['trace_ids']
    else:
        trace_ids = {}
    for i in range(trace_dist.length):
        trace = trace_dist._get_value(i)
        trace_weight = float(trace_dist._get_weight(i))
        trace_str = ''.join([variable.address_base if use_address_base else variable.address for variable in trace.variables])
        if trace_str not in traces:
            if trace_str in trace_ids:
                trace_id = trace_ids[trace_str]
            else:
                trace_id = 'T' + str(len(trace_ids) + 1)
                trace_ids[trace_str] = trace_id
            address_id_sequence = ['START'] + [addresses[variable.address_base if use_address_base else variable.address]['address_id'] for variable in trace.variables] + ['END']
            traces[trace_str] = {'count': 1, 'weight': trace_weight, 'trace_id': trace_id, 'trace': trace, 'address_id_sequence': address_id_sequence}
        else:
            traces[trace_str]['count'] += 1
            traces[trace_str]['weight'] += trace_weight
    traces = OrderedDict(sorted(traces.items(), key=lambda v: v[1]['count'], reverse=True))
    address_ids = [i for i in range(len(addresses))]
    address_weights = []
    for key, value in addresses.items():
        address_weights.append(value['count'])
    address_id_dist = Empirical(address_ids, weights=address_weights, name='Address ID')
    unique_trace_ids = [i for i in range(len(traces))]
    trace_weights = []
    for _, value in traces.items():
        trace_weights.append(value['count'])
    trace_id_dist = Empirical(unique_trace_ids, weights=unique_trace_ids, name='Unique trace ID')
    trace_length_dist = trace_dist.map(lambda trace: trace.length).unweighted().rename('Trace length (all)')
    trace_length_controlled_dist = trace_dist.map(lambda trace: trace.length_controlled).unweighted().rename('Trace length (controlled)')
    trace_execution_time_dist = trace_dist.map(lambda trace: trace.execution_time_sec).unweighted().rename('Trace execution time (s)')
    traces_extra = OrderedDict()
    traces_extra['trace_types'] = len(traces)
    traces_extra['trace_length_min'] = float(trace_length_dist.min)
    traces_extra['trace_length_max'] = float(trace_length_dist.max)
    traces_extra['trace_length_mean'] = float(trace_length_dist.mean)
    traces_extra['trace_length_stddev'] = float(trace_length_dist.stddev)
    traces_extra['trace_length_controlled_min'] = float(trace_length_controlled_dist.min)
    traces_extra['trace_length_controlled_max'] = float(trace_length_controlled_dist.max)
    traces_extra['trace_length_controlled_mean'] = float(trace_length_controlled_dist.mean)
    traces_extra['trace_length_controlled_stddev'] = float(trace_length_controlled_dist.stddev)
    traces_extra['trace_execution_time_min'] = float(trace_execution_time_dist.min)
    traces_extra['trace_execution_time_max'] = float(trace_execution_time_dist.max)
    traces_extra['trace_execution_time_mean'] = float(trace_execution_time_dist.mean)
    traces_extra['trace_execution_time_stddev'] = float(trace_execution_time_dist.stddev)
    return {'traces': traces, 'traces_extra': traces_extra, 'trace_ids': trace_ids, 'address_stats': address_stats, 'trace_id_dist': trace_id_dist, 'trace_length_dist': trace_length_dist, 'trace_length_controlled_dist': trace_length_controlled_dist, 'trace_execution_time_dist': trace_execution_time_dist, 'address_id_dist': address_id_dist}


def trace_histograms(trace_dist, use_address_base=True, figsize=(10, 5), bins=30, plot=False, plot_show=True, file_name=None):
    trace_stats = _trace_stats(trace_dist, use_address_base=use_address_base)
    traces = trace_stats['traces']
    traces_extra = trace_stats['traces_extra']
    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        # mpl.rcParams['font.size'] = 4
        fig, ax = plt.subplots(2, 2, figsize=figsize)

        values = trace_stats['trace_length_dist'].values_numpy()
        weights = trace_stats['trace_length_dist'].weights_numpy()
        name = trace_stats['trace_length_dist'].name
        ax[0, 0].hist(values, weights=weights, density=1, bins=bins)
        ax[0, 0].set_xlabel(name)
        ax[0, 0].set_ylabel('Frequency')
        ax[0, 0].set_yscale('log', nonposy='clip')

        values = trace_stats['trace_length_controlled_dist'].values_numpy()
        weights = trace_stats['trace_length_controlled_dist'].weights_numpy()
        name = trace_stats['trace_length_controlled_dist'].name
        ax[0, 1].hist(values, weights=weights, density=1, bins=bins)
        ax[0, 1].set_xlabel(name)
        # ax[0, 1].set_ylabel('Frequency')
        ax[0, 1].set_yscale('log', nonposy='clip')

        values = trace_stats['address_id_dist'].values_numpy()
        weights = trace_stats['address_id_dist'].weights_numpy()
        name = trace_stats['address_id_dist'].name
        ax[1, 0].hist(values, weights=weights, density=1, bins=len(values))
        ax[1, 0].set_xlabel(name)
        ax[1, 0].set_ylabel('Frequency')
        ax[1, 0].set_yscale('log', nonposy='clip')

        values = trace_stats['trace_execution_time_dist'].values_numpy()
        weights = trace_stats['trace_execution_time_dist'].weights_numpy()
        name = trace_stats['trace_execution_time_dist'].name
        ax[1, 1].hist(values, weights=weights, density=1, bins=bins)
        ax[1, 1].set_xlabel(name)
        # ax[1, 1].set_ylabel('Frequency')
        ax[1, 1].set_yscale('log', nonposy='clip')

        plt.suptitle(trace_dist.name, x=0.0, y=.99, horizontalalignment='left', verticalalignment='top', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if file_name is not None:
            plot_file_name = file_name + '.pdf'
            print('Plotting to file: {}'.format(plot_file_name))
            plt.savefig(plot_file_name)
            report_file_name = file_name + '.txt'
            print('Saving trace report to file: {}'.format(report_file_name))
            with open(report_file_name, 'w') as file:
                file.write('pyprob diagnostics\n')
                for key, value in traces_extra.items():
                    file.write('{}: {}\n'.format(key, value))
            traces_file_name = file_name + '.csv'
            print('Saving traces to file: {}'.format(traces_file_name))
            with open(traces_file_name, 'w') as file:
                file.write('trace_id, count, length, length_controlled, address_id_sequence\n')
                for key, value in traces.items():
                    file.write('{}, {}, {}, {}, {}\n'.format(value['trace_id'], value['count'], len(value['trace'].variables), len(value['trace'].variables_controlled), ' '.join(value['address_id_sequence'])))
        if plot_show:
            plt.show()


def address_histograms(trace_dists, ground_truth_trace=None, figsize=(15, 12), bins=30, use_address_base=True, plot=False, plot_show=True, file_name=None):
    if not isinstance(trace_dists, list):
        trace_dists = [trace_dists]
    dists = {}
    address_stats = None
    address_stats_combined = {}
    for trace_dist in trace_dists:
        print('Collecting values for distribution: {}'.format(trace_dist.name))
        address_stats = _address_stats(trace_dist, use_address_base=use_address_base, reuse_ids_from_address_stats=address_stats)
        addresses = address_stats['addresses']
        for key, val in addresses.items():
            if key in address_stats_combined:
                address_stats_combined[key]['count'] += val['count']
            else:
                address_stats_combined[key] = val
        addresses_extra = address_stats['addresses_extra']
        i = 0
        util.progress_bar_init('Collecting values', len(addresses), 'Addresses')
        for key, value in addresses.items():
            util.progress_bar_update(i)
            i += 1
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
                dist.rename(address_id + '' if variable.name is None else '{} ({})'.format(address_id, variable.name))
                if dist.length == 0:
                    can_render = False
            except Exception:
                can_render = False
            if can_render:
                if key not in dists:
                    dists[key] = {}
                dists[key][trace_dist.name] = dist, variable
        util.progress_bar_end()
    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        mpl.rcParams['font.size'] = 4
        rows, cols = util.tile_rows_cols(len(dists))
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        ax = ax.flatten()
        i = 0
        hist_color_cycle = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'k']))
        hist_colors = {}
        util.progress_bar_init('Plotting histograms', len(dists), 'Histograms')
        for key, value in dists.items():
            util.progress_bar_update(i)
            for trace_dist_name, v in value.items():
                dist = v[0]
                variable = v[1]
                values = dist.values_numpy()
                weights = dist.weights_numpy()
                if trace_dist_name in hist_colors:
                    label = None
                    color = hist_colors[trace_dist_name]
                else:
                    label = trace_dist_name
                    color = hist_color_cycle.pop()
                    hist_colors[trace_dist_name] = color
                if hasattr(variable.distribution, 'low'):
                    range = (float(variable.distribution.low), float(variable.distribution.high))
                else:
                    range = None
                ax[i].hist(values, weights=weights, density=1, bins=bins, color=color, label=label, alpha=0.75, range=range)
                ax[i].set_title(dist.name, fontsize=4, y=0.95)
                ax[i].tick_params(pad=0., length=2)
                # ax[i].set_aspect(aspect='equal', adjustable='box-forced')
                if ground_truth_trace is not None:
                    vline_x = None
                    if use_address_base:
                        address_base = variable.address_base
                        if address_base in ground_truth_trace.variables_dict_address_base:
                            vline_x = float(ground_truth_trace.variables_dict_address_base[address_base].value)
                    else:
                        address = variable.address
                        if address in ground_truth_trace.variables_dict_address:
                            vline_x = float(ground_truth_trace.variables_dict_address[address].value)
                    if vline_x is not None:
                        ax[i].axvline(x=vline_x, linestyle='dashed', color='gray', linewidth=0.75)
            i += 1
        util.progress_bar_end()
        fig.legend()
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=1.5, wspace=0.85)
        if file_name is not None:
            plot_file_name = file_name + '.pdf'
            print('Plotting to file: {}'.format(plot_file_name))
            plt.savefig(plot_file_name)
            report_file_name = file_name + '.txt'
            print('Saving address report to file: {}'.format(report_file_name))
            with open(report_file_name, 'w') as file:
                file.write('pyprob diagnostics\n')
                file.write(('aggregated ' if use_address_base else '') + 'address report\n')
                for key, value in addresses_extra.items():
                    file.write('{}: {}\n'.format(key, value))
            addresses_file_name = file_name + '.csv'
            print('Saving addresses to file: {}'.format(addresses_file_name))
            with open(addresses_file_name, 'w') as file:
                file.write('address_id, count, name, controlled, replaced, observable, observed, {}\n'.format('address_base' if use_address_base else 'address'))
                for key, value in address_stats_combined.items():
                    name = '' if value['variable'].name is None else value['variable'].name
                    file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value['address_id'], value['count'], name, value['variable'].control, value['variable'].replace, value['variable'].observable, value['variable'].observed, key))
        if plot_show:
            plt.show()


def network(inference_network, save_dir=None):
    train_iter_per_sec = inference_network._total_train_iterations / inference_network._total_train_seconds
    train_traces_per_sec = inference_network._total_train_traces / inference_network._total_train_seconds
    train_traces_per_iter = inference_network._total_train_traces / inference_network._total_train_iterations
    train_loss_initial = inference_network._history_train_loss[0]
    train_loss_final = inference_network._history_train_loss[-1]
    train_loss_change = train_loss_final - train_loss_initial
    train_loss_change_per_sec = train_loss_change / inference_network._total_train_seconds
    train_loss_change_per_iter = train_loss_change / inference_network._total_train_iterations
    train_loss_change_per_trace = train_loss_change / inference_network._total_train_traces
    if len(inference_network._history_valid_loss) > 0:
        valid_loss_initial = inference_network._history_valid_loss[0]
        valid_loss_final = inference_network._history_valid_loss[-1]
        valid_loss_change = valid_loss_final - valid_loss_initial
        valid_loss_change_per_sec = valid_loss_change / inference_network._total_train_seconds
        valid_loss_change_per_iter = valid_loss_change / inference_network._total_train_iterations
        valid_loss_change_per_trace = valid_loss_change / inference_network._total_train_traces

    stats = OrderedDict()
    stats['pyprob version'] = __version__
    stats['torch version'] = torch.__version__
    stats['network type'] = inference_network._network_type
    stats['number of parameters'] = inference_network._history_num_params[-1]
    stats['pre-generated layers'] = inference_network._layers_pre_generated
    stats['modified'] = inference_network._modified
    stats['updates'] = inference_network._updates
    stats['trained on device'] = str(inference_network._device)
    stats['distributed training'] = inference_network._distributed_backend is not None
    stats['distributed backend'] = inference_network._distributed_backend
    stats['distributed world size'] = inference_network._distributed_world_size
    stats['optimizer'] = str(inference_network._optimizer_type)
    stats['learning rate'] = inference_network._learning_rate
    stats['momentum'] = inference_network._momentum
    stats['batch size'] = inference_network._batch_size
    stats['total train. seconds'] = inference_network._total_train_seconds
    stats['total train. traces'] = inference_network._total_train_traces
    stats['total train. iterations'] = inference_network._total_train_iterations
    stats['train. iter. per second'] = train_iter_per_sec
    stats['train. traces per second'] = train_traces_per_sec
    stats['train. traces per iter.'] = train_traces_per_iter
    stats['train. loss initial'] = train_loss_initial
    stats['train. loss final'] = train_loss_final
    stats['train. loss change per second'] = train_loss_change_per_sec
    stats['train. loss change per iter.'] = train_loss_change_per_iter
    stats['train. loss change per trace'] = train_loss_change_per_trace
    if len(inference_network._history_valid_loss) > 0:
        stats['valid. loss initial'] = valid_loss_initial
        stats['valid. loss final'] = valid_loss_final
        stats['valid. loss change per second'] = valid_loss_change_per_sec
        stats['valid. loss change per iter.'] = valid_loss_change_per_iter
        stats['valid. loss change per trace'] = valid_loss_change_per_trace

    if save_dir is not None:
        if not os.path.exists(save_dir):
            print('Directory does not exist, creating: {}'.format(save_dir))
            os.makedirs(save_dir)
        file_name_stats = os.path.join(save_dir, 'inference_network_stats.txt')
        print('Saving diagnostics information to {} '.format(file_name_stats))
        with open(file_name_stats, 'w') as file:
            file.write('pyprob diagnostics report\n')
            for key, value in stats.items():
                file.write('{}: {}\n'.format(key, value))
            file.write('architecture:\n')
            file.write(str(next(inference_network.modules())))

        mpl.rcParams['axes.unicode_minus'] = False
        plt.switch_backend('agg')

        file_name_loss = os.path.join(save_dir, 'loss.pdf')
        print('Plotting loss to file: {}'.format(file_name_loss))
        fig = plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        ax.plot(inference_network._history_train_loss_trace, inference_network._history_train_loss, label='Training')
        ax.plot(inference_network._history_valid_loss_trace, inference_network._history_valid_loss, label='Validation')
        ax.legend()
        plt.xlabel('Training traces')
        plt.ylabel('Loss')
        plt.grid()
        fig.tight_layout()
        plt.savefig(file_name_loss)

        file_name_num_params = os.path.join(save_dir, 'num_params.pdf')
        print('Plotting number of parameters to file: {} '.format(file_name_num_params))
        fig = plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        ax.plot(inference_network._history_num_params_trace, inference_network._history_num_params, label='Training')
        plt.xlabel('Training traces')
        plt.ylabel('Number of parameters')
        plt.grid()
        fig.tight_layout()
        plt.savefig(file_name_num_params)

        save_dir_params = os.path.join(save_dir, 'params')
        if not os.path.exists(save_dir_params):
            print('Directory does not exist, creating: {}'.format(save_dir_params))
            os.makedirs(save_dir_params)
        file_name_params = os.path.join(save_dir_params, 'params.csv')
        with open(file_name_params, 'w') as file:
            file.write('file_name, param_name\n')
            num_params = len(list(inference_network.named_parameters()))
            util.progress_bar_init('Plotting inference network parameters', num_params, 'Parameters')
            for index, param in enumerate(inference_network.named_parameters()):
                util.progress_bar_update(index+1)
                print()
                file_name_param = os.path.join(save_dir_params, 'param_{}.png'.format(index))
                param_name = param[0]
                file.write('{}, {}\n'.format(os.path.basename(file_name_param), param_name))
                print('Plotting to file: {}  parameter: {}'.format(file_name_param, param_name))
                param_val = param[1].cpu().detach().numpy()
                if param_val.ndim == 1:
                    param_val = np.expand_dims(param_val, 1)
                elif param_val.ndim > 2:
                    print('Warning: reshaping parameter {} to 2D for plotting.'.format(param_name, param_val.ndim))
                    c = param_val.shape[0]
                    param_val = np.reshape(param_val, (c, -1))
                fig = plt.figure(figsize=(10, 7))
                ax = plt.subplot(111)
                heatmap = ax.pcolor(param_val, cmap=plt.cm.jet)
                ax.invert_yaxis()
                plt.xlabel('{} {}'.format(param_name, param_val.shape))
                plt.colorbar(heatmap)
                # fig.tight_layout()
                plt.savefig(file_name_param)
                plt.close()
            util.progress_bar_end()
    return stats


def graph(trace_dist, use_address_base=True, n_most_frequent=None, base_graph=None, file_name=None, normalize_weights=True):
    graph = Graph(trace_dist=trace_dist, use_address_base=use_address_base, n_most_frequent=n_most_frequent, base_graph=base_graph, normalize_weights=normalize_weights)
    if file_name is not None:
        graph.render_to_file(file_name, background_graph=base_graph)
        for trace_id, trace_graph in graph.trace_graphs():
            trace_graph.render_to_file('{}_{}'.format(file_name, trace_id), background_graph=(graph if base_graph is None else base_graph))
    return graph


def address_dictionary(address_dictionary, file_name):
    print('Saving address_id, address pairs to {}'.format(file_name))
    util.create_path(file_name)
    with open(file_name, 'w') as file:
        file.write('address_id, address\n')
        for key, value in address_dictionary._shelf.items():
            if key.startswith('__id__'):
                address_id = key.replace('__id__', '')
                address = value
                file.write('{}, {}\n'.format(address_id, address))


def log_prob(trace_dists, resolution=1000, names=None, figsize=(10, 5), xlabel="Iteration", ylabel='Log probability', xticks=None, yticks=None, log_xscale=False, log_yscale=False, plot=False, plot_show=True, file_name=None, min_index=None, max_index=None, *args, **kwargs):
    if type(trace_dists) != list:
        raise TypeError('Expecting a list of posterior trace distributions, each from a call to a Model\'s posterior_traces.')
    if min_index is None:
        min_i = 0
    iters = []
    log_probs = []
    for j in range(len(trace_dists)):
        if type(trace_dists[j][0]) != Trace:
            raise TypeError('Expecting a list of posterior trace distributions, each from a call to a Model\'s posterior_traces.')
        if max_index is None:
            max_i = trace_dists[j].length
        else:
            max_i = min(trace_dists[j].length, max_index)
        num_traces = max_i - min_i
        iters.append(list(range(min_i, max_i, max(1, int(num_traces / resolution)))))
        time_start = time.time()
        prev_duration = 0
        len_str_num_traces = len(str(num_traces))
        print('Loading trace log-probabilities to memory')
        print('Time spent  | Time remain.| Progress             | {} | Traces/sec'.format('Trace'.ljust(len_str_num_traces * 2 + 1)))
        vals = []
        for i in iters[j]:
            vals.append(trace_dists[j]._get_value(i).log_prob)
            duration = time.time() - time_start
            if (duration - prev_duration > util._print_refresh_rate) or (i == num_traces - 1):
                prev_duration = duration
                traces_per_second = (i + 1) / duration
                print('{} | {} | {} | {}/{} | {:,.2f}       '.format(util.days_hours_mins_secs_str(duration), util.days_hours_mins_secs_str((num_traces - i) / traces_per_second), util.progress_bar(i+1, num_traces), str(i+1).rjust(len_str_num_traces), num_traces, traces_per_second), end='\r')
                sys.stdout.flush()
        print()
        log_probs.append(vals)

    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        if names is None:
            names = ['{}'.format(trace_dists[i].name) for i in range(len(log_probs))]
        for i in range(len(log_probs)):
            plt.plot(iters[i], log_probs[i], *args, **kwargs, label=names[i])
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        fig.tight_layout()
        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()

    return np.array(iters), np.array(log_probs)


def _n_most_frequent_addresses(trace_dist, n_most_frequent, num_traces=None):
    address_counts = defaultdict(int)
    if num_traces is None:
        num_traces = trace_dist.length
    util.progress_bar_init('Collecting most frequent addresses', num_traces, 'Traces')
    for i in range(num_traces):
        trace = trace_dist._get_value(i)
        util.progress_bar_update(i)
        for variable in trace.variables:
            if variable.value.nelement() == 1:
                address_counts[variable.address] += 1
    util.progress_bar_end()
    address_counts = {k: v for k, v in address_counts.items() if v >= num_traces}
    address_counts = OrderedDict(sorted(address_counts.items(), key=lambda x: x[1], reverse=True))
    ret = []
    for i, address in enumerate(address_counts):
        ret.append(address)
        if i + 1 == n_most_frequent:
            break
    return ret


def _variable_values(trace_dist, names=None, n_most_frequent=None, num_traces=None):
    if num_traces is None:
        num_traces = trace_dist.length
    if names is None:
        name_counts = defaultdict(int)
        for i in range(num_traces):
            trace = trace_dist._get_value(i)
            for name in trace.named_variables.keys():
                name_counts[name] += 1
        names = [name for name in name_counts if name_counts[name] == num_traces]  # Names of named variables that are found in all traces

    variable_values = {}
    # Select named variables to process
    for name in names:
        variable = trace_dist[0].named_variables[name]
        if variable.value.nelement() == 1:
            if variable.address not in variable_values:
                variable_values[variable.address] = {'variable': None, 'values': np.ones(num_traces) * np.nan}
            variable_values[variable.address]['variable'] = variable

    # Select most frequent variables to process (either named or not named)
    if n_most_frequent is not None:
        addresses = _n_most_frequent_addresses(trace_dist, n_most_frequent, num_traces)
        for address in addresses:
            variable = trace_dist[0].variables_dict_address[address]
            if variable.value.nelement() == 1:
                if variable.address not in variable_values:
                    variable_values[variable.address] = {'variable': None, 'values': np.ones(num_traces) * np.nan}
                variable_values[variable.address]['variable'] = variable

    if len(variable_values) == 0:
        raise RuntimeError('No variables with scalar value.')

    util.progress_bar_init('Loading selected variables to memory', num_traces, 'Traces')
    for i in range(num_traces):
        trace = trace_dist._get_value(i)
        util.progress_bar_update(i)
        for address, v in variable_values.items():
            v['values'][i] = float(trace.variables_dict_address[address].value)
    util.progress_bar_end()
    return variable_values


def autocorrelation(trace_dist, names=None, lags=None, n_most_frequent=None, figsize=(10, 5), xticks=None, yticks=None, log_xscale=True, plot=False, plot_show=True, file_name=None, *args, **kwargs):
    if type(trace_dist) != Empirical:
        raise TypeError('Expecting a trace distribution.')
    if type(trace_dist[0]) != Trace:
        raise TypeError('Expecting a trace distribution.')

    def _autocorrelation(values, lags):
        ret = np.array([1. if lag == 0 else np.corrcoef(values[lag:], values[:-lag])[0][1] for lag in lags])
        # nan is encountered when there is no variance in the values, the foloowing might be used to assign autocorrelation of 1 to such cases
        # ret[np.isnan(ret)] = 1.
        return ret

    num_traces = trace_dist.length

    if lags is None:
        lags = np.unique(np.logspace(0, np.log10(num_traces/2)).astype(int))

    variable_values = _variable_values(trace_dist, names, n_most_frequent)
    for i, (address, v) in enumerate(variable_values.items()):
        print('Computing autocorrelation for variable address: {}, name: {} ({} of {})'.format(v['variable'].address, v['variable'].name, i + 1, len(variable_values)))
        v['autocorrelation'] = _autocorrelation(v['values'], lags)

    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        plt.axhline(y=0, linewidth=1, color='black')
        other_legend_added = False
        for address, v in variable_values.items():
            name = v['variable'].name
            autocorrelation = v['autocorrelation']
            if name is None:
                label = None
                if not other_legend_added:
                    label = '{} most frequent addresses'.format(len(variable_values))
                    other_legend_added = True
                plt.plot(lags, autocorrelation, *args, **kwargs, linewidth=1, color='gray', label=label)
            else:
                plt.plot(lags, autocorrelation, *args, **kwargs, label=v['variable'].name)
        if log_xscale:
            plt.xscale('log')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend(loc='best')
        fig.tight_layout()
        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()
    return lags, variable_values


def gelman_rubin(trace_dists, names=None, iters=None, n_most_frequent=50, figsize=(10, 5), xticks=None, yticks=None, log_xscale=False, log_yscale=False, plot=False, plot_show=True, file_name=None, *args, **kwargs):
    def _r_hat(values):
        m, n = values.shape[0], values.shape[1]  # m: number of chains, n: length of chains
        if m < 2:
            raise ValueError('Gelman-Rubin diagnostic requires at least two chains')
        b = n * np.var(np.mean(values, axis=1), axis=0, ddof=1)  # Between-chain variance
        w = np.mean(np.var(values, axis=1, ddof=1), axis=0)  # Within-chain variance
        v_hat = ((n-1) / n) * w + b / n  # Estimate of marginal posterior variance
        r_hat = np.sqrt(v_hat / w)
        return r_hat

    def _r_hats(values, iters):
        ret = np.zeros_like(iters, dtype=float)
        for i, iter in enumerate(iters):
            ret[i] = _r_hat(values[:, :iter])
        return ret

    trace_lengths = [trace.length for trace in trace_dists]
    num_traces = min(trace_lengths)
    if max(trace_lengths) != num_traces:
        print('Distributions have unequal length, setting the length to minimum: {}'.format(num_traces))

    if iters is None:
        iters = np.unique(np.logspace(0, np.log10(num_traces)).astype(int))

    variable_values = {}
    for trace_dist in trace_dists:
        vv = _variable_values(trace_dist, names, n_most_frequent, num_traces)
        for address, v in vv.items():
            if address in variable_values:
                variable_values[address]['values'] = np.vstack((variable_values[address]['values'], v['values']))
            else:
                variable_values[address] = v

    for i, (address, v) in enumerate(variable_values.items()):
        print('Computing R-hat for variable address: {}, name: {} ({} of {})'.format(v['variable'].address, v['variable'].name, i + 1, len(variable_values)))
        v['rhat'] = _r_hats(v['values'], iters)

    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        plt.axhline(y=1, linewidth=1, color='black')
        other_legend_added = False
        for address, v in variable_values.items():
            name = v['variable'].name
            rhat = v['rhat']
            if name is None:
                label = None
                if not other_legend_added:
                    label = '{} most frequent addresses'.format(len(variable_values))
                    other_legend_added = True
                plt.plot(iters, rhat, *args, **kwargs, linewidth=1, color='gray', label=label)
            else:
                plt.plot(iters, rhat, *args, **kwargs, label=v['variable'].name)
        if log_xscale:
            plt.xscale('log')
        if log_yscale:
            plt.yscale('log', nonposy='clip')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel('Iteration')
        plt.ylabel('R-hat')
        plt.legend(loc='best')
        fig.tight_layout()
        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()

    return iters, variable_values


def jensen_shannon(trace_dist_p, trace_dist_q, names=None, n_most_frequent=50,
                   use_address_base=False,
                   figsize=(10, 5), bins=30, xticks=None, yticks=None, log_xscale=False, log_yscale=True, plot=False, plot_show=True, file_name=None,
                   posterior_flag=False):
    def plot_func(variable_info, address_stats_combined,
                  figsize, bins, xticks, yticks, log_xscale, log_yscale, plot_show, file_name):
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        mpl.rcParams['font.size'] = 4
        hist_color_cycle = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'k']))

        # Plot Jensen–Shannon histogram
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_ylabel('#')
        ax1.set_xlabel('Jensen–Shannon')
        ax1.hist([v['divergence'] for v in variable_info.values()], alpha=0.75, color=hist_color_cycle[-1])

        plt.tight_layout()
        if file_name is not None:
            plot_file_name = file_name + '_divergence_hist.pdf'
            print('Plotting to file: {}'.format(plot_file_name))
            plt.savefig(plot_file_name)
        if plot_show:
            plt.show()

        # Plot address histograms
        hist_color_cycle = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'b', 'k']))
        rows, cols = util.tile_rows_cols(len(variable_info))
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        ax = ax.flatten()
        i = 0
        hist_colors = {}
        util.progress_bar_init('Plotting histograms', len(variable_info), 'Histograms')
        for address_id, v in variable_info.items():
            util.progress_bar_update(i)
            for dist_label, values in v['values'].items():
                # dist_label is the main trace distribution name
                if dist_label in hist_colors:
                    label = None
                    color = hist_colors[dist_label]
                else:
                    label = dist_label
                    color = hist_color_cycle.pop()
                    hist_colors[dist_label] = color
                range_ = (np.min(values), np.max(values))
                ax[i].hist(values, density=1, bins=bins, color=color, label=label, alpha=0.75, range=range_)
                ax[i].set_title('{} / {:.2f}'.format(variable_info[address_id]['dist'].name, v['divergence']), fontsize=4, y=0.95)
                ax[i].tick_params(pad=0., length=2)
                # ax[i].set_aspect(aspect='equal', adjustable='box-forced')
            i += 1
        util.progress_bar_end()
        fig.legend()
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=1.5, wspace=0.85)
        if file_name is not None:
            plot_file_name = file_name + '_address.pdf'
            print('Plotting to file: {}'.format(plot_file_name))
            plt.savefig(plot_file_name)
        if plot_show:
            plt.show()

    def _shrink(empirical_dist, final_size, posterior_flag):
        '''
        Given an empirical distribution, shrinks it to final_size.
        If final size is bigger than the current size, it will be unchanged
        If it has unifrom weights, we assume it comes from some sort of MCMC, therefore it will be thinned.
        Otherwise, it will be resampled.
        '''
        if empirical_dist._uniform_weights:
            # Samples are supposedly from MCMC => should thin
            if final_size != len(empirical_dist):
                return empirical_dist.thin(final_size)
            else:
                # No need for thinning if the size is exactly the same
                return empirical_dist
        else:
            # Weighted samples => should resample
            # Even if the size is exactly the same as expected, resampling is needed to make weights uniform
            if posterior_flag:
                # Resample according to proposal sample weights, in order to get a posterior approximation.
                return empirical_dist.resample(final_size)
            else:
                # Randomly choose from the proposal samples and remove the weights.
                return Empirical(np.random.choice(empirical_dist, final_size, replace=False))

    def generate_variable_empiricals(trace_dist, chosen_addresses, use_address_base):
        '''
        Arguments
        ---------
        trace_dist          Empirical distribution over traces
        chosen_addresses    List of chosen addresses (could be address bases, depending on use_address_base)
        num_traces          length of the empirical distribution to consider
        use_address_base    If True, uses address base as variable identifier.

        Returns
        -------
        A dictionary of the same addresses (could be a subset of addresses, if no sample exist for that variable)
        to another dictionary:
            'variable'  ->  variable information for the variable at the address
            'dist'      ->  Empirical distribution over this single variable
        '''
        variable_info = defaultdict(lambda: {'variable': None, 'values': [], 'log_weights': []})
        num_traces = trace_dist.length
        util.progress_bar_init('Loading selected variables to memory', num_traces, 'Traces')
        for i in range(num_traces):
            trace = trace_dist._get_value(i)
            util.progress_bar_update(i)
            trace_weight = trace_dist._get_log_weight(i)
            for address in chosen_addresses:
                # Set the trace dictionary too look for address in. Depending on use_address_base, it could be variables_dict_address or variables_dict_address_base.
                if use_address_base:
                    trace_variables_dict = trace.variables_dict_address_base
                else:
                    trace_variables_dict = trace.variables_dict_address
                if address in trace_variables_dict:
                    if trace_variables_dict[address].value.nelement() == 1:
                        if variable_info[address]['variable'] is None:
                            variable_info[address]['variable'] = trace_variables_dict[address]
                        variable_info[address]['values'].append(trace_variables_dict[address].value)
                        variable_info[address]['log_weights'].append(trace_weight)

        ret_val = {address: {'variable': info['variable'],
                             'dist': Empirical(info['values'], log_weights=info['log_weights'])}
                   for address, info in variable_info.items() if info['variable'] is not None and len(info['values']) > 0}
        util.progress_bar_end()

        return ret_val

    def _n_most_frequent_from_stats(address_stats, n):
        addresses = address_stats['addresses']
        ordered_addresses = OrderedDict(sorted(addresses.items(), key=lambda x: x[1]['count'], reverse=True))
        res = []
        i = 0
        for candidate_address, variable_info in ordered_addresses.items():
            # variable_info is the value associated with 'addresses' key in address_stats (has ['count', 'weight', 'address_id', 'variable'] as keys)
            if len(res) >= n:
                break
            if variable_info['variable'].value.nelement() == 1:
                # Ignore multi-dimensional random variables
                res.append(candidate_address)
        return res

    def get_renamed_variable_empiricals(trace_dists, names, n_most_frequent, use_address_base):
        '''
        Arguments
        ---------
        trace_dists         List of trace distributions
        names               List of chosen variable names (if any)
        n_most_frequent     Number of most frequent variables to include in the result
        use_address_base    If True, uses address base as variable identifier.

        Returns
        -------
        variable_empiricals A dictionary from variable address_id to:
                            A dictionary with the keys ['dist', 'variable']
                            The value associated with 'dist' is the distribution over
                            the corresponding variable, renamed to be used as plot title
        address_stats       Combined address_stats for trace distributions in the given trace_dists.

        '''
        variable_empiricals = []
        address_stats = None
        for trace_dist in trace_dists:
            print('Computing address stats for {}'.format(trace_dist.name))
            address_stats = _address_stats(trace_dist, use_address_base=use_address_base, reuse_ids_from_address_stats=address_stats)
            chosen_addresses = _n_most_frequent_from_stats(address_stats, n_most_frequent)
            variable_empirical = generate_variable_empiricals(trace_dist, chosen_addresses, use_address_base)
            # Rename variable_empirical keys from address (or address_base) to address_id:
            keys = list(variable_empirical.keys())
            for k in keys:
                new_key = address_stats['address_ids'][k]
                variable_empirical[new_key] = variable_empirical.pop(k)

            # Rename the distributions based on their assigned address_id:
            for address_id, variable_info in variable_empirical.items():
                # variable_info is a dictionary with ['variable', 'dist'] as keys
                variable = variable_info['variable']
                variable_info['dist'].rename(address_id + '' if variable.name is None else '{} ({})'.format(address_id, variable.name))

            variable_empiricals.append(variable_empirical)
        return variable_empiricals, address_stats


    '''
    Arguments
    ---------
    posterior_flag          If true, will calculate Jensen–Shannon divergence with the approximated posterior
                            rather than the proposal. It is done by resampling proposal samples.

    Returns
    -------
    variable_info           A dictionary of sample address_ids to their analysis.
                            The analysis itself is a dictronary with the
                            following specifications: (key -> value)
                            'variable' -> an object of type Variable containing
                                          the random variable specifications.
                            'values' -> A dictionary from trace name (has input trace names as keys)
                                        to numpy array of sampled values in P distribution
                                        (shape: nxd, where n is the number of samples and d is the size of each sampled value)
                            'divergence' -> The Jensen–Shannon divergence.
    '''
    assert isinstance(bins, int)
    [variable_empirical_p, variable_empirical_q], address_stats_combined = get_renamed_variable_empiricals([trace_dist_p, trace_dist_q], names, n_most_frequent, use_address_base)

    variable_info = {}      # Will contain the output  i.e. divergence and sample values for all addresses.
    common_address_ids = set(variable_empirical_p.keys()) & set(variable_empirical_q.keys()) #address_id_to_variable
    util.progress_bar_init('Computing Jensen-Shannon divergence', len(common_address_ids), 'Variables')
    for i, address_id in enumerate(common_address_ids):
        util.progress_bar_update(i)
        variable_info_log = 'address: {}, name: {} ({} of {})'.format(variable_empirical_p[address_id]['variable'].address, variable_empirical_p[address_id]['variable'].name, i + 1, len(common_address_ids))
        var_dist_p = variable_empirical_p[address_id]['dist']
        var_dist_q = variable_empirical_q[address_id]['dist'] # Get the distribution for the same variable from the other distribution
        num_samples = min(len(var_dist_p), len(var_dist_q))
        if num_samples < 10:
            print('\nToo few samples for {} ({}). Skipping...'.format(address_id, num_samples))
            continue
        '''
        # There is no need for having the same number of samples to compute Jensen–Shannon.
        var_dist_p = _shrink(var_dist_p, num_samples, posterior_flag)
        var_dist_q = _shrink(var_dist_q, num_samples, posterior_flag)

        assert var_dist_p._uniform_weights
        assert var_dist_q._uniform_weights
        '''

        v_p = var_dist_p.values_numpy()
        v_q = var_dist_q.values_numpy()

        range_ = (min(np.min(v_p), np.min(v_q)), max(np.max(v_p), np.max(v_q)))
        bins_seq = np.linspace(*range_, bins)
        bin_width = bins_seq[1] - bins_seq[0]
        p_probs = np.histogram(v_p, bins=bins, density=True)[0]
        q_probs = np.histogram(v_q, bins=bins, density=True)[0]

        # add a small amount to all porbs so that nothing is zero. It avoids problems in computing Jensen–Shannon.
        p_probs += 1e-20 / bins
        q_probs += 1e-20 / bins

        p_categorical = torch.distributions.categorical.Categorical(probs=util.to_tensor(p_probs))
        q_categorical = torch.distributions.categorical.Categorical(probs=util.to_tensor(q_probs))

        kl_pq = kl_divergence(p_categorical, q_categorical).item()
        kl_qp = kl_divergence(q_categorical, p_categorical).item()
        divergence = (kl_pq + kl_qp) / 2

        # Aggregate info about P and Q in "variable info"
        v = {}
        v['divergence'] = divergence
        for vv_key in variable_empirical_p[address_id]:
            if vv_key != 'values':
                v[vv_key] = variable_empirical_p[address_id][vv_key]
        v['values'] = {}
        v['values'][trace_dist_p.name] = v_p
        v['values'][trace_dist_q.name] = v_q
        v['bin_width'] = bin_width
        variable_info[address_id] = v

    util.progress_bar_end()

    if plot:
        plot_func(variable_info, address_stats_combined,
                 figsize, bins, xticks, yticks, log_xscale, log_yscale, plot_show, file_name)

    if file_name is not None:
        divergence_info_csv = file_name + '_info.csv'
        print('Saving Jensen–Shannon diagnostic info to CSV: {}'.format(divergence_info_csv))
        with open(divergence_info_csv, 'w') as csvfile:
            csv_titles = ['Name', 'ID', 'Address', 'Divergence', 'sample-size', 'bin width']
            csv_writer = csv.DictWriter(csvfile, fieldnames=csv_titles)
            csv_writer.writeheader()
            for k, v in variable_info.items():
                info = {}
                info['Name'] = v['variable'].name
                info['ID'] = k
                info['Address'] = v['variable'].address
                info['Divergence'] = v['divergence']
                info['sample-size'] = len(v['values'][next(iter(v['values']))])
                info['bin width'] = v['bin_width']
                csv_writer.writerow(info)

            divergence_values = [v['divergence'] for v in variable_info.values()]
            divergence_mean = np.mean(divergence_values)
            divergence_var = np.var(divergence_values)
            csv_writer.writerow({csv_titles[0]: 'Number of bins', csv_titles[1]: bins})
            csv_writer.writerow({csv_titles[0]: 'Jensen–Shannon mean', csv_titles[1]: divergence_mean})
            csv_writer.writerow({csv_titles[0]: 'Jensen–Shannon variance', csv_titles[1]: divergence_var})
            print('Jensen–Shannon mean = {}, Jensen–Shannon variance = {}'.format(divergence_mean, divergence_var))

    return variable_info
