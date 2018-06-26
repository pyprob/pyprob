import os
import gc
import sys
import datetime
import torch
from collections import Counter, OrderedDict
from itertools import islice
import numpy as np
import re
from pylatex import Document, Section, Subsection, Subsubsection, Command, Figure, SubFigure, Tabularx, LongTable, FootnoteText, FlushLeft, TextColor
from pylatex.utils import italic, NoEscape
import matplotlib
matplotlib.use('Agg') # Do not use X server
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams['axes.axisbelow'] = True
matplotlib.rcParams['figure.dpi'] = 600
import seaborn as sns
sns.set_style("ticks")
colors = ["dark grey", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))
import matplotlib.pyplot as plt
import pydotplus
from io import BytesIO
import matplotlib.image as mpimg
from PIL import Image

from . import util, __version__


def save_report(model, file_name, detailed_traces=2):
    print('Saving analytics report to {}.tex and {}.pdf'.format(file_name, file_name))

    inference_network = model._inference_network
    iter_per_sec = inference_network._total_train_iterations / inference_network._total_train_seconds
    traces_per_sec = inference_network._total_train_traces / inference_network._total_train_seconds
    traces_per_iter = inference_network._total_train_traces / inference_network._total_train_iterations
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

    sys.stdout.write('Generating report...                                           \r')
    sys.stdout.flush()

    geometry_options = {'tmargin':'1.5cm', 'lmargin':'1cm', 'rmargin':'1cm', 'bmargin':'1.5cm'}
    doc = Document('basic', geometry_options=geometry_options)
    doc.preamble.append(NoEscape(r'\usepackage[none]{hyphenat}'))
    doc.preamble.append(NoEscape(r'\usepackage{float}'))
    # doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\ttdefault}'))

    doc.preamble.append(Command('title', 'pyprob analytics: ' + model.name))
    doc.preamble.append(Command('date', NoEscape(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))))
    doc.append(NoEscape(r'\maketitle'))
    # doc.append(NoEscape(r'\small'))

    print('Analytics: Current system')
    with doc.create(Section('Current system', numbering=False)):
        with doc.create(Tabularx('ll')) as table:
            table.add_row(('pyprob version', __version__))
            table.add_row(('PyTorch version', torch.__version__))

    # doc.append(NoEscape(r'\newpage'))
    print('Analytics: Inference network')
    with doc.create(Section('Inference network', numbering=False)):
        print('Analytics: Inference network.File')
        with doc.create(Section('File')):
            with doc.create(Tabularx('ll')) as table:
                # table.add_row(('File name', file_name))
                # file_size = '{:,}'.format(os.path.getsize(file_name))
                # table.add_row(('File size', file_size + ' Bytes'))
                table.add_row(('Created', inference_network._created))
                table.add_row(('Modified', inference_network._modified))
                table.add_row(('Updates to file', inference_network._updates))
        print('Analytics: Inference network.Training')
        with doc.create(Section('Training')):
            with doc.create(Tabularx('ll')) as table:
                table.add_row(('pyprob version', inference_network._pyprob_version))
                table.add_row(('PyTorch version', inference_network._torch_version))
                table.add_row(('Trained on', inference_network._trained_on))
                table.add_row(('Total training time', '{0}'.format(util.days_hours_mins_secs_str(inference_network._total_train_seconds))))
                table.add_row(('Total training traces', '{:,}'.format(inference_network._total_train_traces)))
                table.add_row(('Traces / s', '{:,.2f}'.format(traces_per_sec)))
                table.add_row(('Traces / iteration', '{:,.2f}'.format(traces_per_iter)))
                table.add_row(('Iterations', '{:,}'.format(inference_network._total_train_iterations)))
                table.add_row(('Iterations / s', '{:,.2f}'.format(iter_per_sec)))
                table.add_row(('Optimizer', inference_network._optimizer_type))
                table.add_row(('Validation set size', inference_network._valid_batch.length))
        print('Analytics: Inference network.Training loss')
        with doc.create(Subsection('Training loss')):
            with doc.create(Tabularx('ll')) as table:
                table.add_row(('Initial loss', '{:+.6e}'.format(train_loss_initial)))
                table.add_row(('Final loss', '{:+.6e}'.format(train_loss_final)))
                table.add_row(('Loss change / s', '{:+.6e}'.format(train_loss_change_per_sec)))
                table.add_row(('Loss change / iteration', '{:+.6e}'.format(train_loss_change_per_iter)))
                table.add_row(('Loss change / trace', '{:+.6e}'.format(train_loss_change_per_trace)))
        print('Analytics: Inference network.Validation loss')
        with doc.create(Subsection('Validation loss')):
            with doc.create(Tabularx('ll')) as table:
                table.add_row(('Initial loss', '{:+.6e}'.format(valid_loss_initial)))
                table.add_row(('Final loss', '{:+.6e}'.format(valid_loss_final)))
                table.add_row(('Loss change / s', '{:+.6e}'.format(valid_loss_change_per_sec)))
                table.add_row(('Loss change / iteration', '{:+.6e}'.format(valid_loss_change_per_iter)))
                table.add_row(('Loss change / trace', '{:+.6e}'.format(valid_loss_change_per_trace)))
        with doc.create(Figure(position='H')) as plot:
            fig = plt.figure(figsize=(10,6))
            ax = plt.subplot(111)
            ax.plot(inference_network._history_train_loss_trace, inference_network._history_train_loss, label='Training')
            ax.plot(inference_network._history_valid_loss_trace, inference_network._history_valid_loss, label='Validation')
            ax.legend()
            plt.xlabel('Training traces')
            plt.ylabel('Loss')
            plt.grid()
            fig.tight_layout()
            plot.add_plot(width=NoEscape(r'\textwidth'))
            plot.add_caption('Loss plot.')

        print('Analytics: Inference network.Neural network modules')
        with doc.create(Section('Neural network modules')):
            with doc.create(Tabularx('ll')) as table:
                table.add_row(('Total trainable parameters', '{:,}'.format(inference_network._history_num_params[-1])))
                # table.add_row(('Softmax boost', inference_network.softmax_boost))
                # table.add_row(('Dropout', inference_network.dropout))
                # table.add_row(('Standardize inputs', inference_network.standardize_observes))
            with doc.create(Figure(position='H')) as plot:
                fig = plt.figure(figsize=(10,4))
                ax = plt.subplot(111)
                ax.plot(inference_network._history_num_params_trace, inference_network._history_num_params)
                plt.xlabel('Training traces')
                plt.ylabel('Number of parameters')
                plt.grid()
                fig.tight_layout()
                plot.add_plot(width=NoEscape(r'\textwidth'))
                plot.add_caption('Number of parameters.')

            doc.append(NoEscape(r'\newpage'))
            print('Analytics: Inference network.Neural network modules.All modules')
            with doc.create(Subsection('All modules')):
                doc.append(str(inference_network))

            for m_name, m in inference_network.named_modules():
                if (m_name != ''):
                    regex = r'(sample_embedding_layer\(\S*\)._)|(proposal_layer\(\S*\)._)|(_observe_embedding_layer.)|(_lstm)'
                    if len(list(re.finditer(regex, m_name))) > 0:
                    # if ('_observe_embedding_layer.' in m_name) or ('sample_embedding_layer.' in m_name) or ('proposal_layer.' in m_name):
                        doc.append(NoEscape(r'\newpage'))
                        with doc.create(Subsubsection(m_name)):
                            doc.append(str(m))
                            for p_name, p in m.named_parameters():
                                if not 'bias' in p_name:
                                    with doc.create(Figure(position='H')) as plot:
                                        fig = plt.figure(figsize=(10,10))
                                        ax = plt.subplot(111)
                                        plt.imshow(np.transpose(util.weights_to_image(p),(1,2,0)), interpolation='none')
                                        plt.axis('off')
                                        plot.add_plot(width=NoEscape(r'\textwidth'))
                                        plot.add_caption(m_name + '_' + p_name)
            # doc.append(NoEscape(r'\newpage'))
            # print('Analytics: Inference network.Neural network modules.Address embeddings')
            # with doc.create(Subsection('Address embeddings')):
            #     for p_name, p in inference_network.named_parameters():
            #         if ('address_embedding' in p_name):
            #             with doc.create(Figure(position='H')) as plot:
            #                 fig = plt.figure(figsize=(10,10))
            #                 ax = plt.subplot(111)
            #                 plt.imshow(np.transpose(util.weights_to_image(p),(1,2,0)), interpolation='none')
            #                 plt.axis('off')
            #                 plot.add_plot(width=NoEscape(r'\textwidth'))
            #                 plot.add_caption(FootnoteText(p_name.replace('::', ':: ')))

        gc.collect()
        doc.append(NoEscape(r'\newpage'))
        print('Analytics: Inference network.Traces')
        with doc.create(Section('Traces')):
            with doc.create(Tabularx('ll')) as table:
                table.add_row(('Total training traces', '{:,}'.format(inference_network._total_train_traces)))
            print('Analytics: Inference network.Traces.Distributions encountered')
            with doc.create(Subsection('Distributions encountered')):
                with doc.create(Tabularx('ll')) as table:
                    # print([v[2] for v in inference_network._address_stats.values()])
                    distributions = set([v[2] for v in inference_network._address_stats.values()])
                    num_distributions = len(distributions)
                    table.add_row(('Number of distributions', num_distributions))
                    table.add_empty_row()
                    for distribution in distributions:
                        table.add_row((distribution, ''))
            print('Analytics: Inference network.Traces.Addresses encountered')
            with doc.create(Subsection('Addresses encountered')):
                with doc.create(Tabularx('lX')) as table:
                    num_addresses_all = len(inference_network._address_stats.keys())
                    table.add_row(('Number of addresses', num_addresses_all))
                    num_addresses_controlled = len([k for k, v in inference_network._address_stats.items() if v[3]])
                    num_addresses_replaced = len([k for k, v in inference_network._address_stats.items() if v[3] and v[4]])
                    num_addresses_observed = len([k for k, v in inference_network._address_stats.items() if v[5]])
                    table.add_row((TextColor('red', 'Number of addresses (controlled)'), TextColor('red', num_addresses_controlled)))
                    table.add_row((TextColor('green', 'Number of addresses (replaced)'), TextColor('green', num_addresses_replaced)))
                    table.add_row((TextColor('blue', 'Number of addresses (observed)'), TextColor('blue', num_addresses_observed)))
                    table.add_row(('Number of addresses (uncontrolled)', num_addresses_all - num_addresses_controlled - num_addresses_observed))
                    table.add_empty_row()
                doc.append('\n')
                with doc.create(LongTable('llllllp{12cm}')) as table:
                    # table.add_empty_row()
                    table.add_row(FootnoteText('Count'), FootnoteText('ID'), FootnoteText('Distrib.'), FootnoteText('Ctrl.'), FootnoteText('Replace'), FootnoteText('Obs.'), FootnoteText('Address'))
                    table.add_hline()

                    # address_to_abbrev = {}
                    # abbrev_to_address =
                    # abbrev_i = 0
                    # sorted_addresses = sorted(inference_network.address_histogram.items(), key=lambda x:x[1], reverse=True)
                    plt_all_addresses = []
                    plt_all_counts = []
                    plt_all_colors = []
                    plt_controlled_addresses = []
                    plt_controlled_counts = []
                    plt_controlled_colors = []
                    address_id_to_count = {}
                    address_id_to_color = {}
                    address_id_count_total = 0
                    for address, vals in inference_network._address_stats.items():
                        address = address.replace('::', ':: ')
                        count = vals[0]
                        address_id = vals[1]
                        distribution = vals[2]
                        control = vals[3]
                        replace = vals[4]
                        observed = vals[5]
                        plt_all_addresses.append(address_id)
                        plt_all_counts.append(1 if replace else count)
                        address_id_to_count[address_id] = count
                        address_id_count_total += count
                        if control:
                            if replace:
                                color = 'green'
                                plt_controlled_counts.append(1)
                            else:
                                color = 'red'
                                plt_controlled_counts.append(count)

                            plt_controlled_addresses.append(address_id)
                            plt_controlled_colors.append(color)
                        elif observed:
                            color = 'blue'
                            plt_controlled_addresses.append(address_id)
                            plt_controlled_colors.append(color)
                            plt_controlled_counts.append(count)
                        else:
                            color = 'black'
                        table.add_row((TextColor(color, FootnoteText('{:,}'.format(count))), TextColor(color, FootnoteText(address_id)), TextColor(color, FootnoteText(distribution)), TextColor(color, FootnoteText(control)), TextColor(color, FootnoteText(replace)), TextColor(color, FootnoteText(observed)), TextColor(color, FootnoteText(address))))
                        plt_all_colors.append(color)
                        address_id_to_color[address_id] = color

                gc.collect()
                with doc.create(Figure(position='H')) as plot:
                    fig = plt.figure(figsize=(10, 5))
                    ax = plt.subplot(111)
                    plt_x = range(len(plt_all_addresses))
                    ax.bar(plt_x, plt_all_counts, color=plt_all_colors)
                    plt.xticks(plt_x, plt_all_addresses)
                    plt.xlabel('Address ID')
                    plt.ylabel('Count')
                    # plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Histogram of all addresses. Red: controlled, green: replaced, black: uncontrolled, blue: observed.')

                with doc.create(Figure(position='H')) as plot:
                    fig = plt.figure(figsize=(10, 5))
                    ax = plt.subplot(111)
                    plt_x = range(len(plt_controlled_addresses))
                    ax.bar(plt_x, plt_controlled_counts, color=plt_controlled_colors)
                    plt.xticks(plt_x, plt_controlled_addresses)
                    plt.xlabel('Address ID')
                    plt.ylabel('Count')
                    # plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Histogram of controlled and observed addresses. Red: controlled, green: replaced, blue: observed.')

            gc.collect()
            print('Analytics: Inference network.Traces.Trace lengths')
            with doc.create(Subsection('Trace lengths')):
                with doc.create(Tabularx('ll')) as table:
                    trace_lengths_controlled = [v[3] for v in inference_network._trace_stats.values()]
                    trace_lengths_controlled_min = min(trace_lengths_controlled)
                    trace_lengths_controlled_max = max(trace_lengths_controlled)
                    trace_lengths_all = [v[2] for v in inference_network._trace_stats.values()]
                    trace_lengths_all_min = min(trace_lengths_all)
                    trace_lengths_all_max = max(trace_lengths_all)
                    s_controlled = 0
                    s_all = 0
                    total_count = 0
                    for _, v in inference_network._trace_stats.items():
                        trace_length_controlled = v[3]
                        trace_length_all = v[2]
                        count = v[0]
                        s_controlled += trace_length_controlled * count
                        total_count += count
                        s_all += trace_length_all * count
                    trace_length_controlled_mean = s_controlled / total_count
                    trace_length_all_mean = s_all / total_count
                    table.add_row(('Trace length min', '{:,}'.format(trace_lengths_all_min)))
                    table.add_row(('Trace length max', '{:,}'.format(trace_lengths_all_max)))
                    table.add_row(('Trace length mean', '{:.2f}'.format(trace_length_all_mean)))
                    table.add_row(('Controlled trace length min', '{:,}'.format(trace_lengths_controlled_min)))
                    table.add_row(('Controlled trace length max', '{:,}'.format(trace_lengths_controlled_max)))
                    table.add_row(('Controlled trace length mean', '{:.2f}'.format(trace_length_controlled_mean)))
                with doc.create(Figure(position='H')) as plot:
                    plt_counter = dict(Counter(trace_lengths_all))
                    plt_lengths = [i for i in range(0, trace_lengths_all_max + 1)]
                    plt_counts = [plt_counter[i] if i in plt_counter else 0 for i in range(0, trace_lengths_all_max + 1)]
                    fig = plt.figure(figsize=(10,5))
                    ax = plt.subplot(111)
                    ax.bar(plt_lengths, plt_counts, width=trace_lengths_all_max/500.)
                    plt.xlabel('Length')
                    plt.ylabel('Count')
                    # plt.yscale('log')
                    # plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Histogram of trace lengths.')
                with doc.create(Figure(position='H')) as plot:
                    plt_counter = dict(Counter(trace_lengths_controlled))
                    plt_lengths = [i for i in range(0, trace_lengths_controlled_max + 1)]
                    plt_counts = [plt_counter[i] if i in plt_counter else 0 for i in range(0, trace_lengths_controlled_max + 1)]
                    fig = plt.figure(figsize=(10,5))
                    ax = plt.subplot(111)
                    ax.bar(plt_lengths, plt_counts)
                    plt.xlabel('Length')
                    plt.ylabel('Count')
                    # plt.yscale('log')
                    # plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Histogram of controlled trace lengths.')

            gc.collect()
            print('Analytics: Inference network.Traces.Unique traces encountered')
            with doc.create(Subsection('Unique traces encountered')):
                detailed_traces = min(len(inference_network._trace_stats), detailed_traces)
                with doc.create(Tabularx('ll')) as table:
                    table.add_row(('Unique traces encountered', '{:,}'.format(len(inference_network._trace_stats))))
                    table.add_row(('Unique traces rendered in detail', '{:,}'.format(detailed_traces)))
                doc.append('\n')
                with doc.create(LongTable('llllp{15cm}')) as table:
                    # table.add_empty_row()
                    table.add_row(FootnoteText('Count'), FootnoteText('ID'), FootnoteText('Len.'), FootnoteText('Ctrl. len.'), FootnoteText('Unique trace'))
                    table.add_hline()

                    plt_traces = []
                    plt_counts = []
                    for trace_str, vals in inference_network._trace_stats.items():
                        count = vals[0]
                        trace_id = vals[1]
                        length_all = vals[2]
                        length_controlled = vals[3]
                        addresses_controlled = vals[4]
                        addresses_controlled_str = ' '.join(addresses_controlled)
                        plt_traces.append(trace_id)
                        plt_counts.append(count)
                        table.add_row((FootnoteText('{:,}'.format(count)), FootnoteText(trace_id), FootnoteText('{:,}'.format(length_all)), FootnoteText('{:,}'.format(length_controlled)), FootnoteText(addresses_controlled_str)))

                with doc.create(Figure(position='H')) as plot:
                    fig = plt.figure(figsize=(10, 5))
                    ax = plt.subplot(111)
                    plt_x = range(len(plt_traces))
                    ax.bar(plt_x, plt_counts)
                    plt.xticks(plt_x, plt_traces)
                    plt.xlabel('Trace ID')
                    plt.ylabel('Count')
                    # plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Histogram of unique traces.')

                with doc.create(Figure(position='H')) as plot:
                    sorted_trace_stats = OrderedDict(sorted(dict(inference_network._trace_stats).items(), key=lambda x: x[1], reverse=True))
                    master_trace_pairs = {}
                    transition_count_total = 0
                    for trace_str, vals in sorted_trace_stats.items():
                        count = vals[0]
                        ta = vals[4]
                        for left, right in zip(ta, ta[1:]):
                            if (left, right) in master_trace_pairs:
                                master_trace_pairs[(left, right)] += count
                            else:
                                master_trace_pairs[(left, right)] = count
                            transition_count_total += count
                    fig = plt.figure(figsize=(10, 5))
                    ax = plt.subplot(111)
                    master_graph = pydotplus.graphviz.Dot(graph_type='digraph', rankdir='LR')
                    transition_count_max = 0
                    for p, count in master_trace_pairs.items():
                        if count > transition_count_max:
                            transition_count_max = count
                        nodes = master_graph.get_node(p[0])
                        if len(nodes) > 0:
                            n0 = nodes[0]
                        else:
                            n0 = pydotplus.Node(p[0])
                            master_graph.add_node(n0)
                        nodes = master_graph.get_node(p[1])
                        if len(nodes) > 0:
                            n1 = nodes[0]
                        else:
                            n1 = pydotplus.Node(p[1])
                            master_graph.add_node(n1)
                        master_graph.add_edge(pydotplus.Edge(n0, n1, weight=count))
                    for node in master_graph.get_nodes():
                        node.set_color('gray')
                        node.set_fontcolor('gray')
                    for edge in master_graph.get_edges():
                        edge.set_color('gray')

                    master_graph_annotated = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())
                    for node in master_graph_annotated.get_nodes():
                        # color = util.rgb_to_hex(util.rgb_blend((1, 1, 1), (1, 0, 0), address_id_to_count[node.obj_dict['name']] / address_id_count_total))
                        address_id = node.obj_dict['name']
                        node.set_style('filled')
                        node.set_fillcolor(address_id_to_color[address_id])
                        node.set_color('black')
                        node.set_fontcolor('black')
                    for edge in master_graph_annotated.get_edges():
                        (left, right) = edge.obj_dict['points']
                        count = master_trace_pairs[(left, right)]
                        edge.set_label(count)
                        # color = util.rgb_to_hex((1.5*(count/transition_count_total), 0, 0))
                        edge.set_color('black')
                        edge.set_penwidth(2.5 * count / transition_count_max)

                    png_str = master_graph_annotated.create_png(prog=['dot', '-Gsize=90', '-Gdpi=600'])
                    bio = BytesIO()
                    bio.write(png_str)
                    bio.seek(0)
                    img = np.asarray(mpimg.imread(bio))
                    plt.imshow(util.crop_image(img), interpolation='bilinear')
                    plt.axis('off')
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Succession of controlled addresses (accumulated over all traces). Red: controlled, green: replaced, blue: observed.')

                for trace_str, vals in OrderedDict(islice(sorted_trace_stats.items(), 0, detailed_traces)).items():
                    count = vals[0]
                    trace_id = vals[1]
                    doc.append(NoEscape(r'\newpage'))
                    with doc.create(Subsubsection('Unique trace ' + trace_id)):
                        sys.stdout.write('Rendering unique trace {0}...                                       \r'.format(trace_id))
                        sys.stdout.flush()

                        addresses = len(plt_controlled_addresses)
                        trace_addresses = vals[4]

                        with doc.create(Tabularx('ll')) as table:
                            table.add_row(FootnoteText('Count'), FootnoteText('{:,}'.format(count)))
                            table.add_row(FootnoteText('Controlled length'), FootnoteText('{:,}'.format(len(trace_addresses))))
                        doc.append('\n')

                        im = np.zeros((addresses, len(trace_addresses)))
                        for i in range(len(trace_addresses)):
                            address = trace_addresses[i]
                            address_i = plt_controlled_addresses.index(address)
                            im[address_i, i] = 1
                        truncate = 100
                        for col_start in range(0, len(trace_addresses), truncate):
                            col_end = min(col_start + truncate, len(trace_addresses))
                            with doc.create(Figure(position='H')) as plot:
                                fig = plt.figure(figsize=(20 * ((col_end + 4 - col_start) / truncate),4))
                                ax = plt.subplot(111)
                                # ax.imshow(im,cmap=plt.get_cmap('Greys'))
                                sns.heatmap(im[:, col_start:col_end], cbar=False, linecolor='lightgray', linewidths=.5, cmap='Greys', yticklabels=plt_controlled_addresses, xticklabels=np.arange(col_start, col_end))
                                plt.yticks(rotation=0)
                                fig.tight_layout()
                                plot.add_plot(width=NoEscape(r'{0}\textwidth'.format((col_end + 4 - col_start) / truncate)), placement=NoEscape(r'\raggedright'))

                        with doc.create(Figure(position='H')) as plot:
                            pairs = {}
                            for left, right in zip(trace_addresses, trace_addresses[1:]):
                                if (left, right) in pairs:
                                    pairs[(left, right)] += 1
                                else:
                                    pairs[(left, right)] = 1

                            fig = plt.figure(figsize=(10, 5))
                            ax = plt.subplot(111)
                            graph = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())

                            trace_address_to_count = {}
                            for address in trace_addresses:
                                if address in trace_address_to_count:
                                    trace_address_to_count[address] += 1
                                else:
                                    trace_address_to_count[address] = 1

                            transition_count_max = 0
                            for p, count in pairs.items():
                                if count > transition_count_max:
                                    transition_count_max = count
                                left_node = graph.get_node(p[0])[0]
                                right_node = graph.get_node(p[1])[0]
                                edge = graph.get_edge(p[0], p[1])[0]

                                # color = util.rgb_to_hex(util.rgb_blend((1,1,1), (1,0,0), trace_address_to_count[p[0]] / len(trace_addresses)))
                                left_node.set_style('filled')
                                left_node.set_fillcolor(address_id_to_color[p[0]])
                                left_node.set_color('black')
                                left_node.set_fontcolor('black')

                                # color = util.rgb_to_hex(util.rgb_blend((1,1,1), (1,0,0), trace_address_to_count[p[0]] / len(trace_addresses)))
                                right_node.set_style('filled')
                                right_node.set_fillcolor(address_id_to_color[p[1]])
                                right_node.set_color('black')
                                right_node.set_fontcolor('black')

                                # (left, right) = edge.obj_dict['points']
                                edge.set_label(count)
                                # color = util.rgb_to_hex((1.5*(count/len(trace_addresses)),0,0))
                                edge.set_color('black')
                                edge.set_penwidth(2.5 * count / transition_count_max)

                            png_str = graph.create_png(prog=['dot', '-Gsize=90', '-Gdpi=600'])
                            bio = BytesIO()
                            bio.write(png_str)
                            bio.seek(0)
                            img = np.asarray(mpimg.imread(bio))
                            plt.imshow(util.crop_image(img), interpolation='bilinear')
                            plt.axis('off')
                            plot.add_plot(width=NoEscape(r'\textwidth'))
                            plot.add_caption('Succession of controlled addresses (for one trace of type ' + trace_id + '). Red: controlled, green: replaced, blue: observed.')

                        with doc.create(Tabularx('lp{16cm}')) as table:
                            table.add_row(FootnoteText('Trace'), FootnoteText(' '.join(trace_addresses)))



    doc.generate_pdf(file_name, clean_tex=False)
    sys.stdout.write('                                                               \r')
    sys.stdout.flush()
