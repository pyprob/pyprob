import torch
import os
from collections import OrderedDict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import __version__, util, PriorInflation, InferenceEngine
from .distributions import Empirical
from .graph import Graph


class Analytics():
    def __init__(self, model):
        self._model = model

    def inference_network(self, report_dir=None):
        if self._model._inference_network is None:
            raise RuntimeError('The model does not have a trained inference network. Use learn_inference_network first.')
        return self.network_statistics(self._model._inference_network, report_dir)

    def prior_graph(self, num_traces=1000, prior_inflation=PriorInflation.DISABLED, use_address_base=True, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, report_dir=None, *args, **kwargs):
        trace_dist = self._model.prior_traces(num_traces=num_traces, prior_inflation=prior_inflation, *args, **kwargs)
        if report_dir is not None:
            report_dir = os.path.join(report_dir, 'prior')
            if not os.path.exists(report_dir):
                print('Directory does not exist, creating: {}'.format(report_dir))
                os.makedirs(report_dir)
            file_name_dist = os.path.join(report_dir, 'traces.distribution')
            print('Saving trace distribution to {} ...'.format(file_name_dist))
            trace_dist.save(file_name_dist)
        return self.graph(trace_dist, use_address_base, n_most_frequent, base_graph, report_dir, bins, log_xscale, log_yscale)

    def posterior_graph(self, num_traces=1000, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, observe=None, use_address_base=True, bins=100, log_xscale=False, log_yscale=False, n_most_frequent=None, base_graph=None, report_dir=None, *args, **kwargs):
        trace_dist = self._model.posterior_traces(num_traces=num_traces, inference_engine=inference_engine, observe=observe, *args, **kwargs)
        if report_dir is not None:
            report_dir = os.path.join(report_dir, 'posterior/' + inference_engine.name)
            if not os.path.exists(report_dir):
                print('Directory does not exist, creating: {}'.format(report_dir))
                os.makedirs(report_dir)
            file_name_dist = os.path.join(report_dir, 'traces.distribution')
            print('Saving trace distribution to {} ...'.format(file_name_dist))
            trace_dist.save(file_name_dist)
        return self.graph(trace_dist, use_address_base, n_most_frequent, base_graph, report_dir, bins, log_xscale, log_yscale)

    @staticmethod
    def network_statistics(inference_network, report_dir=None):
        train_iter_per_sec = inference_network._total_train_iterations / inference_network._total_train_seconds
        train_traces_per_sec = inference_network._total_train_traces / inference_network._total_train_seconds
        train_traces_per_iter = inference_network._total_train_traces / inference_network._total_train_iterations
        train_loss_initial = inference_network._history_train_loss[0]
        train_loss_final = inference_network._history_train_loss[-1]
        train_loss_change = train_loss_final - train_loss_initial
        train_loss_change_per_sec = train_loss_change / inference_network._total_train_seconds
        train_loss_change_per_iter = train_loss_change / inference_network._total_train_iterations
        train_loss_change_per_trace = train_loss_change / inference_network._total_train_traces
        valid_loss_initial = inference_network._history_valid_loss[0]
        valid_loss_final = inference_network._history_valid_loss[-1]
        valid_loss_change = valid_loss_final - valid_loss_initial
        valid_loss_change_per_sec = valid_loss_change / inference_network._total_train_seconds
        valid_loss_change_per_iter = valid_loss_change / inference_network._total_train_iterations
        valid_loss_change_per_trace = valid_loss_change / inference_network._total_train_traces

        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['modified'] = inference_network._modified
        stats['updates'] = inference_network._updates
        stats['trained_on_device'] = str(inference_network._device)
        stats['valid_batch_size'] = inference_network._valid_batch_size
        stats['total_train_seconds'] = inference_network._total_train_seconds
        stats['total_train_traces'] = inference_network._total_train_traces
        stats['total_train_iterations'] = inference_network._total_train_iterations
        stats['train_iter_per_sec'] = train_iter_per_sec
        stats['train_traces_per_sec'] = train_traces_per_sec
        stats['train_traces_per_iter'] = train_traces_per_iter
        stats['train_loss_initial'] = train_loss_initial
        stats['train_loss_final'] = train_loss_final
        stats['train_loss_change_per_sec'] = train_loss_change_per_sec
        stats['train_loss_change_per_iter'] = train_loss_change_per_iter
        stats['train_loss_change_per_trace'] = train_loss_change_per_trace
        stats['valid_loss_initial'] = valid_loss_initial
        stats['valid_loss_final'] = valid_loss_final
        stats['valid_loss_change_per_sec'] = valid_loss_change_per_sec
        stats['valid_loss_change_per_iter'] = valid_loss_change_per_iter
        stats['valid_loss_change_per_trace'] = valid_loss_change_per_trace

        if report_dir is not None:
            if not os.path.exists(report_dir):
                print('Directory does not exist, creating: {}'.format(report_dir))
                os.makedirs(report_dir)
            file_name_stats = os.path.join(report_dir, 'inference_network_stats.txt')
            print('Saving analytics information to {} ...'.format(file_name_stats))
            with open(file_name_stats, 'w') as file:
                file.write('pyprob analytics report\n')
                for key, value in stats.items():
                    file.write('{}: {}\n'.format(key, value))
                file.write('architecture:\n')
                file.write(str(next(inference_network.modules())))

            mpl.rcParams['axes.unicode_minus'] = False
            plt.switch_backend('agg')

            file_name_loss = os.path.join(report_dir, 'loss.pdf')
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

            file_name_num_params = os.path.join(report_dir, 'num_params.pdf')
            print('Plotting number of parameters to file: {} ...'.format(file_name_num_params))
            fig = plt.figure(figsize=(10, 7))
            ax = plt.subplot(111)
            ax.plot(inference_network._history_num_params_trace, inference_network._history_num_params, label='Training')
            plt.xlabel('Training traces')
            plt.ylabel('Number of parameters')
            plt.grid()
            fig.tight_layout()
            plt.savefig(file_name_num_params)

            report_dir_params = os.path.join(report_dir, 'params')
            if not os.path.exists(report_dir_params):
                print('Directory does not exist, creating: {}'.format(report_dir_params))
                os.makedirs(report_dir_params)
            for param in inference_network.named_parameters():
                param_name = param[0]
                file_name_param = os.path.join(report_dir_params, param_name + '.png')
                print('Plotting parameter to file: {} ...'.format(file_name_param))
                param_val = param[1].detach().numpy()
                if param_val.ndim == 1:
                    param_val = np.expand_dims(param_val, 1)
                if param_val.ndim > 2:
                    print('Cannot render parameter {} because it is {}-dimensional.'.format(param_name, param_val.ndim))
                else:
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    heatmap = ax.pcolor(param_val, cmap=plt.cm.jet)
                    ax.invert_yaxis()
                    plt.xlabel('{} {}'.format(param_name, param_val.shape))
                    plt.colorbar(heatmap)
                    # fig.tight_layout()
                    plt.savefig(file_name_param)
        return stats

    @staticmethod
    def graph(trace_dist, use_address_base=True, n_most_frequent=None, base_graph=None, report_dir=None, bins=100, log_xscale=False, log_yscale=False):
        stats = OrderedDict()
        stats['pyprob_version'] = __version__
        stats['torch_version'] = torch.__version__
        stats['num_distribution_elements'] = len(trace_dist)

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
        trace_length_dist = trace_dist.map(lambda trace: trace.length).unweighted().rename('Trace length (all)')
        trace_length_controlled_dist = trace_dist.map(lambda trace: trace.length_controlled).unweighted().rename('Trace length (controlled)')
        trace_execution_time_dist = trace_dist.map(lambda trace: trace.execution_time_sec).unweighted().rename('Trace execution time (s)')

        stats['trace_types'] = len(trace_stats)
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
            if not os.path.exists(report_dir):
                print('Directory does not exist, creating: {}'.format(report_dir))
                os.makedirs(report_dir)
            file_name_stats = os.path.join(report_dir, 'stats.txt')
            print('Saving analytics information to {} ...'.format(file_name_stats))
            with open(file_name_stats, 'w') as file:
                file.write('pyprob analytics report\n')
                for key, value in stats.items():
                    file.write('{}: {}\n'.format(key, value))

            file_name_addresses = os.path.join(report_dir, 'addresses.csv')
            print('Saving addresses to {} ...'.format(file_name_addresses))
            with open(file_name_addresses, 'w') as file:
                file.write('address_id, weight, name, controlled, replaced, observable, observed, {}\n'.format('address_base' if use_address_base else 'address'))
                for key, value in address_stats.items():
                    name = '' if value['variable'].name is None else value['variable'].name
                    file.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(value['address_id'], value['weight'], name, value['variable'].control, value['variable'].replace, value['variable'].observable, value['variable'].observed, key))

            file_name_traces = os.path.join(report_dir, 'traces.csv')
            print('Saving addresses to {} ...'.format(file_name_traces))
            with open(file_name_traces, 'w') as file:
                file.write('trace_id, weight, length, length_controlled, address_id_sequence\n')
                for key, value in trace_stats.items():
                    file.write('{}, {}, {}, {}, {}\n'.format(value['trace_id'], value['weight'], len(value['trace'].variables), len(value['trace'].variables_controlled), ' '.join(value['address_id_sequence'])))

            file_name_address_id_dist = os.path.join(report_dir, 'address_ids.pdf')
            print('Saving trace type distribution to {} ...'.format(file_name_address_id_dist))
            address_id_dist.plot_histogram(bins=range(len(address_stats)), xticks=range(len(address_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_address_id_dist)

            file_name_trace_id_dist = os.path.join(report_dir, 'trace_ids.pdf')
            print('Saving trace type distribution to {} ...'.format(file_name_trace_id_dist))
            trace_id_dist.plot_histogram(bins=range(len(trace_stats)), xticks=range(len(trace_stats)), log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_id_dist)

            file_name_trace_length_dist = os.path.join(report_dir, 'trace_length_all.pdf')
            print('Saving trace length (all) distribution to {} ...'.format(file_name_trace_length_dist))
            trace_length_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_dist)

            file_name_trace_length_controlled_dist = os.path.join(report_dir, 'trace_length_controlled.pdf')
            print('Saving trace length (controlled) distribution to {} ...'.format(file_name_trace_length_controlled_dist))
            trace_length_controlled_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_length_controlled_dist)

            file_name_trace_execution_time_dist = os.path.join(report_dir, 'trace_execution_time.pdf')
            print('Saving trace execution time distribution to {} ...'.format(file_name_trace_execution_time_dist))
            trace_execution_time_dist.plot_histogram(bins=bins, log_xscale=log_xscale, log_yscale=log_yscale, color='black', show=False, file_name=file_name_trace_execution_time_dist)

            report_latent_root = os.path.join(report_dir, 'latent_structure')
            if not os.path.exists(report_latent_root):
                print('Directory does not exist, creating: {}'.format(report_latent_root))
                os.makedirs(report_latent_root)
            file_name_latent_structure_all_pdf = os.path.join(report_latent_root, 'latent_structure_all')
            print('Rendering latent structure graph (all) to {} ...'.format(file_name_latent_structure_all_pdf))
            if base_graph is None:
                master_graph.render_to_file(file_name_latent_structure_all_pdf)
            else:
                master_graph.render_to_file(file_name_latent_structure_all_pdf, background_graph=base_graph)

            for i in range(len(trace_stats)):
                trace_id = list(trace_stats.values())[i]['trace_id']
                file_name_latent_structure = os.path.join(report_latent_root, 'latent_structure_most_freq_{}_{}'.format(i+1, trace_id))
                print('Saving latent structure graph {} of {} to {} ...'.format(i+1, len(trace_stats), file_name_latent_structure))
                graph = master_graph.get_sub_graph(i)
                if base_graph is None:
                    graph.render_to_file(file_name_latent_structure, background_graph=master_graph)
                else:
                    graph.render_to_file(file_name_latent_structure, background_graph=base_graph)

            print('Rendering distributions...')
            report_distribution_root = os.path.join(report_dir, 'distributions')
            if not os.path.exists(report_distribution_root):
                print('Directory does not exist, creating: {}'.format(report_distribution_root))
                os.makedirs(report_distribution_root)

            i = 0
            for key, value in address_stats.items():
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

                    dist.rename(address_id + '' if variable.name is None else '{} (name: {})'.format(address_id, variable.name))
                    if dist.length == 0:
                        can_render = False
                except:
                    can_render = False

                if can_render:
                    file_name_dist = os.path.join(report_distribution_root, '{}_distribution.pdf'.format(address_id))
                    print('Saving distribution {} of {} to {} ...'.format(i, len(address_stats), file_name_dist))
                    dist.plot_histogram(bins=bins, color='black', show=False, file_name=file_name_dist)
                    if not dist._uniform_weights:
                        file_name_dist = os.path.join(report_distribution_root, '{}_{}_distribution.pdf'.format(address_id, 'proposal'))
                        print('Saving distribution {} of {} to {} ...'.format(i, len(address_stats), file_name_dist))
                        dist.unweighted().plot_histogram(bins=bins, color='black', show=False, file_name=file_name_dist)

                else:
                    print('Cannot render histogram for {} because it is not scalar valued. Example value: {}'.format(address_id, variable.value))

        return master_graph, stats
