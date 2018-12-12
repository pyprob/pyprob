import torch
import os
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys

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
                            address_id = address_base[2:] + '__' + ('replaced' if variable.replace else str(variable.instance))
                        else:
                            if address_base not in address_base_ids:
                                address_base_id = 'A' + str(len(address_base_ids) + 1)
                                address_base_ids[address_base] = address_base_id
                            address_id = address_base_ids[address_base] + '__' + ('replaced' if variable.replace else str(variable.instance))
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
        address_stats_extra = addresses['address_stats_extra']
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
                ax[i].hist(values, weights=weights, density=1, bins=bins, color=color, label=label, alpha=0.8, range=range)
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
                for key, value in address_stats_extra.items():
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
        print('Saving diagnostics information to {} ...'.format(file_name_stats))
        with open(file_name_stats, 'w') as file:
            file.write('pyprob diagnostics report\n')
            for key, value in stats.items():
                file.write('{}: {}\n'.format(key, value))
            file.write('architecture:\n')
            file.write(str(next(inference_network.modules())))

        mpl.rcParams['axes.unicode_minus'] = False
        plt.switch_backend('agg')

        file_name_loss = os.path.join(save_dir, 'loss.pdf')
        print('Plotting loss to file: {} ...'.format(file_name_loss))
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
        print('Plotting number of parameters to file: {} ...'.format(file_name_num_params))
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
                print('Plotting to file: {}  parameter: {} ...'.format(file_name_param, param_name))
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


def graph(trace_dist, use_address_base=True, n_most_frequent=None, base_graph=None, file_name=None):
    graph = Graph(trace_dist=trace_dist, use_address_base=use_address_base, n_most_frequent=n_most_frequent, base_graph=base_graph)
    if file_name is not None:
        graph.render_to_file(file_name, background_graph=base_graph)
        for trace_id, trace_graph in graph.trace_graphs():
            trace_graph.render_to_file('{}_{}'.format(file_name, trace_id), background_graph=(graph if base_graph is None else base_graph))
    return graph


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
        print('Loading trace log-probabilities to memory...')
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
            print('Plotting to file {} ...'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()

    return np.array(iters), np.array(log_probs)


def autocorrelations(trace_dist, names=None, lags=None, n_most_frequent=None, figsize=(10, 5), xlabel="Lag", ylabel='Autocorrelation', xticks=None, yticks=None, log_xscale=True, plot=False, plot_show=True, file_name=None, *args, **kwargs):
    if type(trace_dist) != Empirical:
        raise TypeError('Expecting a posterior trace distribution, from a call to a Model\'s posterior_traces.')
    if type(trace_dist[0]) != Trace:
        raise TypeError('Expecting a posterior trace distribution, from a call to a Model\'s posterior_traces.')

    def autocorrelation(values, lags):
        ret = np.array([1. if lag == 0 else np.corrcoef(values[lag:], values[:-lag])[0][1] for lag in lags])
        # nan is encountered when there is no variance in the values, the foloowing might be used to assign autocorrelation of 1 to such cases
        # ret[np.isnan(ret)] = 1.
        return ret

    if lags is None:
        lags = np.unique(np.logspace(0, np.log10(trace_dist.length/2)).astype(int))
    variable_values = OrderedDict()
    if names is None:
        for name, variable in trace_dist[-1].named_variables.items():
            if not variable.observed and variable.value.nelement() == 1:
                variable_values[(variable.address, name)] = np.zeros(trace_dist.length)
    else:
        for name in names:
            address = trace_dist[-1].named_variables[name].address
            variable_values[(address, name)] = np.zeros(trace_dist.length)

    if n_most_frequent is not None:
        address_counts = {}
        num_traces = trace_dist.length
        util.progress_bar_init('Collecting most frequent addresses...', num_traces)
        for i in range(num_traces):
            util.progress_bar_update(i)
            trace = trace_dist._get_value(i)
            for variable in trace.variables_controlled:
                if variable.value.nelement() == 1:
                    address = variable.address
                    if address not in address_counts:
                        address_counts[address] = 1
                    else:
                        address_counts[address] += 1
        address_counts = {k: v for k, v in address_counts.items() if v >= num_traces}
        address_counts = OrderedDict(sorted(address_counts.items(), key=lambda x: x[1], reverse=True))
        all_variables_count = 0
        for address, count in address_counts.items():
            variable_values[(address, None)] = np.zeros(trace_dist.length)
            all_variables_count += 1
            if all_variables_count == n_most_frequent:
                break
        print()

    if len(variable_values) == 0:
        raise RuntimeError('No variables with scalar value have been selected.')

    variable_values = OrderedDict(sorted(variable_values.items(), reverse=True))

    num_traces = trace_dist.length
    util.progress_bar_init('Loading selected variables to memory...', num_traces)
    for i in range(num_traces):
        trace = trace_dist._get_value(i)
        for (address, name), values in variable_values.items():
            values[i] = float(trace.variables_dict_address[address].value)
        util.progress_bar_update(i)
    print()
    variable_autocorrelations = {}
    i = 0
    for (address, name), values in variable_values.items():
        i += 1
        print('Computing autocorrelation for variable name: {} ({} of {})...'.format(name, i, len(variable_values)))
        variable_autocorrelations[address] = autocorrelation(values, lags)
    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        plt.axhline(y=0, linewidth=1, color='black')
        other_legend_added = False
        for (address, name), values in variable_values.items():
            if name is None:
                label = None
                if not other_legend_added:
                    label = '{} most frequent addresses'.format(len(variable_values))
                    other_legend_added = True
                plt.plot(lags, variable_autocorrelations[address], *args, **kwargs, linewidth=1, color='gray', label=label)
            else:
                plt.plot(lags, variable_autocorrelations[address], *args, **kwargs, label=name)
        if log_xscale:
            plt.xscale('log')
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.xticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        fig.tight_layout()
        if file_name is not None:
            print('Plotting to file {} ...'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()
    return lags, variable_autocorrelations


def gelman_rubin(trace_dists, names=None, n_most_frequent=None, figsize=(10, 5), xlabel="Iteration", ylabel='R-hat', xticks=None, yticks=None, log_xscale=False, log_yscale=True, plot=False, plot_show=True, file_name=None, *args, **kwargs):
    def merge_dicts(d1, d2):
        for k, v in d2.items():
            if k in d1:
                d1[k] = np.vstack((d1[k], v))
            else:
                d1[k] = v
        return d1

    def gelman_rubin_diagnostic(x, mu=None):
        '''
        Notes
        -----
        The diagnostic is computed by:  math:: \hat{R} = \frac{\hat{V}}{W}

        where :math:`W` is the within-chain variance and :math:`\hat{V}` is
        the posterior variance estimate for the pooled traces.

        :param x: samples
        :param mu, var: true posterior mean and variance; if None, Monte Carlo estimates
        :param logger: None
        :return: r_hat

        References
        ----------
        Gelman et. al. (2012). ‘Bayesian Data Analysis, Third Edition’
        Brooks and Gelman (1998)
        '''
        m, n = x.shape[0], x.shape[1]
        if m < 2:
            raise ValueError(
                'Gelman-Rubin diagnostic requires multiple chains '
                'of the same length.')
        theta = np.mean(x, axis=1)
        sigma = np.var(x, axis=1, ddof=1)
        # theta_m = np.mean(theta, axis=0)
        theta_m = mu if mu else np.mean(theta, axis=0)

        # Calculate between-chain variance
        b = float(n) / float(m-1) * np.sum((theta - theta_m) ** 2)
        # Calculate within-chain variance
        w = 1. / float(m) * np.sum(sigma, axis=0)
        # Estimate of marginal posterior variance
        v_hat = float(n-1) / float(n) * w + b / float(n)
        r_hat = np.sqrt(v_hat / w)
        # logger.info('R: max [%f] min [%f]' % (np.max(r_hat), np.min(r_hat)))
        return r_hat

    def rhat(values, iters, num_traces):
        ret = np.zeros_like(iters, dtype=float)
        num_missing_samples = num_traces - values.shape[1]
        for i, t in enumerate(iters):
            ret[i] = np.nan if t <= num_missing_samples else gelman_rubin_diagnostic(values[:, :t-num_missing_samples])
        # nan is encountered when there is no variance in the values, the following might be used to assign autocorrelation of 1 to such cases
        # ret[np.isnan(ret)] = 1.
        # nan is also injected when the values length is less than the trace_dist length
        return ret

    def single_trace_dist_values(trace_dist, num_traces):
        if type(trace_dist) != Empirical:
            raise TypeError('Expecting an MCMC posterior trace distribution, from a call to posterior_traces with an MCMC inference engine.')
        if type(trace_dist[0]) != Trace:
            raise TypeError('Expecting an MCMC posterior trace distribution, from a call to posterior_traces with an MCMC inference engine.')

        variable_values = {}

        util.progress_bar_init('Loading selected variables to memory...', num_traces)
        for i in range(num_traces):
            trace = trace_dist._get_value(i)
            name_list = trace.named_variables.keys() if names is None else names
            for name in name_list:
                if name not in trace.named_variables:
                    # This random variable is not sampled in the ith trace
                    continue
                variable = trace.named_variables[name]
                if not variable.control and variable.value.nelement() == 1:
                    address = variable.address
                    if (address, name) not in variable_values:
                        # This is the first trace this random variable sample appeared in
                        # Initialize values as a vector of nans. nan means the random variable is not appeared
                        variable_values[(address, name)] = np.ones(num_traces) * np.nan
                    variable_values[(address, name)][i] = float(trace.named_variables[name].value)
            util.progress_bar_update(i)
        print()

        if n_most_frequent is not None:
            address_counts = {}
            util.progress_bar_init('Collecting most frequent addresses...', num_traces)
            for i in range(num_traces):
                util.progress_bar_update(i)
                trace = trace_dist._get_value(i)
                for variable in trace.variables_controlled:
                    if variable.value.nelement() == 1:
                        address = variable.address
                        if address not in address_counts:
                            address_counts[address] = 1
                        else:
                            address_counts[address] += 1
            address_counts = {k: v for k, v in address_counts.items() if v >= num_traces}
            address_counts = OrderedDict(sorted(address_counts.items(), key=lambda x: x[1], reverse=True))
            all_variables_count = 0
            for address, count in address_counts.items():
                variable_values[(address, None)] = np.ones(num_traces) * np.nan
                all_variables_count += 1
                if all_variables_count == n_most_frequent:
                    break
            print()
            # TODO: populate values variable_values[(address, name)][i] = float(trace.named_variables[name].value)
            util.progress_bar_init('Collecting most frequent addresses...', num_traces)
            for i in range(num_traces):
                util.progress_bar_update(i)
                trace = trace_dist._get_value(i)
                for (address, name), value in variable_values.items():
                    variable_values[(address, name)][i] = float(trace.variables_dict_address[address].value)
            print()
        variable_values = OrderedDict(sorted(variable_values.items(), reverse=True))
        return variable_values

    variable_values = {}
    trace_lengths = [trace.length for trace in trace_dists]
    num_traces = min(trace_lengths)
    if max(trace_lengths) != num_traces:
        print('Distributions have unequal length, setting the length to minimum: {}'.format(num_traces))

    for trace in trace_dists:
        variable_values = merge_dicts(variable_values, single_trace_dist_values(trace, num_traces))

    iters = np.unique(np.logspace(0, np.log10(num_traces)).astype(int))

    variable_values = {k: v for k, v in variable_values.items() if v.size == num_traces * (len(trace_dists))}
    # Fill in the spots where a random variable sample is missing
    # and remove all the values before its first appearance in all chains.
    for (address, name), value in variable_values.items():
        x = np.where(~np.isnan(value)) # Find all nans i.e. missing random variable samples
        r, c = x
        first_non_nans = [np.min(c[r == i]) for i in range(value.shape[0]) if i in r] # For each chain, find the first non-nan value
        starting_col = max(first_non_nans) if first_non_nans else value.shape[1]+1 # Set the starting timestep for all chains
                                                                                   # i.e. the first time it is sampled in all chains
        if starting_col != 0:
            # Remove the initial nans
            value = value[:, starting_col:]
            variable_values[(address, name)] = value

        #assert trace_dists[0].length == value.shape[1] + starting_col
        # Fill in the remaining nans with the last value appeared before them
        for chain_idx in range(len(trace_dists)):
            last_value = value[chain_idx, 0]
            #assert not np.isnan(last_value)
            for i in range(value.shape[1]):
                if np.isnan(value[chain_idx, i]):
                    value[chain_idx, i] = last_value
                last_value = value[chain_idx, i]

    variable_rhats = {}
    i = 0
    for (address, name), values in variable_values.items():
        i += 1
        print('Computing R-hat for named variable {} ({} of {})...'.format(name, i, len(variable_values)))
        variable_rhats[address] = rhat(values, iters, num_traces)

    if plot:
        if not plot_show:
            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')
        fig = plt.figure(figsize=figsize)
        plt.axhline(y=1, linewidth=1, color='black')
        other_legend_added = False
        for (address, name), values in variable_values.items():
            if name is None:
                label = None
                if not other_legend_added:
                    label = '{} most frequent addresses'.format(len(variable_values))
                    other_legend_added = True
                plt.plot(iters, variable_rhats[address], *args, **kwargs, linewidth=1, color='gray', label=label)
            else:
                plt.plot(iters, variable_rhats[address], *args, **kwargs, label=name)
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
            print('Plotting to file {} ...'.format(file_name))
            plt.savefig(file_name)
        if plot_show:
            plt.show()

    return iters, variable_rhats
