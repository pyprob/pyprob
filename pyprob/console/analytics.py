#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util, logger
import torch
import argparse
from termcolor import colored
import datetime
import sys
import os
import matplotlib
matplotlib.use('Agg') # Do not use X server
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rcParams['axes.axisbelow'] = True
import seaborn as sns
sns.set_style("ticks")
colors = ["dark grey", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest
import csv
import traceback
from pylatex import Document, Section, Subsection, Subsubsection, Command, Figure, SubFigure, Tabularx, LongTable, FootnoteText, FlushLeft
from pylatex.utils import italic, NoEscape
import pydotplus
from io import BytesIO
import matplotlib.image as mpimg
from PIL import Image

util.logger = logger.Logger('{0}/{1}'.format('.', 'pyprob-log' + util.get_time_stamp()))

def main():
    try:
        parser = argparse.ArgumentParser(description='pyprob ' + pyprob.__version__ + ' (Analytics)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for loading artifacts and saving logs', default='.')
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=123, type=int)
        parser.add_argument('--structure', help='show extra information about artifact structure', action='store_true')
        parser.add_argument('--saveReport', help='save a full analytics report (tex and pdf)', type=str)
        parser.add_argument('--maxTraces', help='maximum number of unique traces to plot in the full analytics report', default=20, type=int)
        parser.add_argument('--saveLoss', help='save training and validation loss history (csv)', type=str)
        parser.add_argument('--saveAddresses', help='save histogram of addresses (csv)', type=str)
        parser.add_argument('--saveTraceLengths', help='save histogram of trace lengths (csv)', type=str)
        opt = parser.parse_args()

        if opt.version:
            print(pyprob.__version__)
            quit()

        util.set_random_seed(opt.seed)
        util.set_cuda(opt.cuda, opt.device)

        util.logger.reset()
        util.logger.log_config()

        file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'pyprob-artifact'), -1)
        util.logger.log(colored('Resuming previous artifact: {}'.format(file_name), 'blue', attrs=['bold']))
        artifact = util.load_artifact(file_name, util.cuda_enabled, util.cuda_device)

        util.logger.log(artifact.get_info())
        util.logger.log()

        if opt.structure:
            util.logger.log()
            util.logger.log(colored('Artifact structure', 'blue', attrs=['bold']))
            util.logger.log()

            util.logger.log(artifact.get_structure_str())
            util.logger.log(artifact.get_parameter_str())

        if opt.saveLoss:
            util.logger.log('Saving training and validation loss history to file: ' + opt.saveLoss)
            with open(opt.saveLoss, 'w') as f:
                data = [artifact.train_history_trace, artifact.train_history_loss, artifact.valid_history_trace, artifact.valid_history_loss]
                writer = csv.writer(f)
                writer.writerow(['train_trace', 'train_loss', 'valid_trace', 'valid_loss'])
                for values in zip_longest(*data):
                    writer.writerow(values)

        if opt.saveAddresses:
            util.logger.log('Saving address histogram to file: ' + opt.saveAddresses)
            with open(opt.saveAddresses, 'w') as f:
                data_count = []
                data_address = []
                data_abbrev = []
                abbrev_i = 0
                for address, count in sorted(artifact.address_histogram.items(), key=lambda x:x[1], reverse=True):
                    abbrev_i += 1
                    data_abbrev.append('A' + str(abbrev_i))
                    data_address.append(address)
                    data_count.append(count)
                data = [data_count, data_abbrev, data_address]
                writer = csv.writer(f)
                writer.writerow(['count', 'unique_address_id','full_address'])
                for values in zip_longest(*data):
                    writer.writerow(values)

        if opt.saveTraceLengths:
            util.logger.log('Saving trace length histogram to file: ' + opt.saveTraceLengths)
            with open(opt.saveTraceLengths, 'w') as f:
                data_trace_length = []
                data_count = []
                for trace_length in artifact.trace_length_histogram:
                    data_trace_length.append(trace_length)
                    data_count.append(artifact.trace_length_histogram[trace_length])
                data = [data_trace_length, data_count]
                writer = csv.writer(f)
                writer.writerow(['trace_length', 'count'])
                for values in zip_longest(*data):
                    writer.writerow(values)

        if opt.saveReport:
            util.logger.log('Saving analytics report to files: ' + opt.saveReport + '.tex and ' + opt.saveReport + '.pdf')

            iter_per_sec = artifact.total_iterations / artifact.total_training_seconds
            traces_per_sec = artifact.total_traces / artifact.total_training_seconds
            traces_per_iter = artifact.total_traces / artifact.total_iterations
            train_loss_initial = artifact.train_history_loss[0]
            train_loss_final = artifact.train_history_loss[-1]
            train_loss_change = train_loss_final - train_loss_initial
            train_loss_change_per_sec = train_loss_change / artifact.total_training_seconds
            train_loss_change_per_iter = train_loss_change / artifact.total_iterations
            train_loss_change_per_trace = train_loss_change / artifact.total_traces
            valid_loss_initial = artifact.valid_history_loss[0]
            valid_loss_final = artifact.valid_history_loss[-1]
            valid_loss_change = valid_loss_final - valid_loss_initial
            valid_loss_change_per_sec = valid_loss_change / artifact.total_training_seconds
            valid_loss_change_per_iter = valid_loss_change / artifact.total_iterations
            valid_loss_change_per_trace = valid_loss_change / artifact.total_traces

            sys.stdout.write('Generating report...                                           \r')
            sys.stdout.flush()

            geometry_options = {'tmargin':'1.5cm', 'lmargin':'1cm', 'rmargin':'1cm', 'bmargin':'1.5cm'}
            doc = Document('basic', geometry_options=geometry_options)
            doc.preamble.append(NoEscape(r'\usepackage[none]{hyphenat}'))
            doc.preamble.append(NoEscape(r'\usepackage{float}'))
            # doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\ttdefault}'))

            doc.preamble.append(Command('title', 'Inference Compilation Analytics'))
            doc.preamble.append(Command('date', NoEscape(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
            doc.append(NoEscape(r'\maketitle'))
            # doc.append(NoEscape(r'\small'))

            with doc.create(Section('Current system')):
                with doc.create(Tabularx('ll')) as table:
                    table.add_row(('pyprob version', pyprob.__version__))
                    table.add_row(('PyTorch version', torch.__version__))

            # doc.append(NoEscape(r'\newpage'))
            with doc.create(Section('Artifact')):
                with doc.create(Subsection('File')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('File name', file_name))
                        file_size = '{:,}'.format(os.path.getsize(file_name))
                        table.add_row(('File size', file_size + ' Bytes'))
                        table.add_row(('Created', artifact.created))
                        table.add_row(('Modified', artifact.modified))
                        table.add_row(('Updates to file', artifact.updates))
                with doc.create(Subsection('Training system')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('pyprob version', artifact.code_version))
                        table.add_row(('PyTorch version', artifact.pytorch_version))
                        table.add_row(('Trained on', artifact.trained_on))
                with doc.create(Subsection('Neural network')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Trainable parameters', '{:,}'.format(artifact.num_params_history_num_params[-1])))
                        table.add_row(('Softmax boost', artifact.softmax_boost))
                        table.add_row(('Dropout', artifact.dropout))
                        table.add_row(('Standardize inputs', artifact.standardize_observes))
                    with doc.create(Figure(position='H')) as plot:
                        fig = plt.figure(figsize=(10,4))
                        ax = plt.subplot(111)
                        ax.plot(artifact.num_params_history_trace, artifact.num_params_history_num_params)
                        plt.xlabel('Training traces')
                        plt.ylabel('Number of parameters')
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Number of parameters.')

                    for m_name, m in artifact.named_modules():
                        if not ('.' in m_name or m_name == ''):
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


            doc.append(NoEscape(r'\newpage'))
            with doc.create(Section('Training')):
                with doc.create(Tabularx('ll')) as table:
                    table.add_row(('Total training time', '{0}'.format(util.days_hours_mins_secs(artifact.total_training_seconds))))
                    table.add_row(('Total training traces', '{:,}'.format(artifact.total_traces)))
                    table.add_row(('Traces / s', '{:,.2f}'.format(traces_per_sec)))
                    table.add_row(('Traces / iteration', '{:,.2f}'.format(traces_per_iter)))
                    table.add_row(('Iterations', '{:,}'.format(artifact.total_iterations)))
                    table.add_row(('Iterations / s', '{:,.2f}'.format(iter_per_sec)))
                    table.add_row(('Optimizer', artifact.optimizer))
                    table.add_row(('Validation set size', artifact.valid_size))

                with doc.create(Subsection('Training loss')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Initial loss', '{:+.6e}'.format(train_loss_initial)))
                        table.add_row(('Final loss', '{:+.6e}'.format(train_loss_final)))
                        table.add_row(('Loss change / s', '{:+.6e}'.format(train_loss_change_per_sec)))
                        table.add_row(('Loss change / iteration', '{:+.6e}'.format(train_loss_change_per_iter)))
                        table.add_row(('Loss change / trace', '{:+.6e}'.format(train_loss_change_per_trace)))
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
                    ax.plot(artifact.train_history_trace, artifact.train_history_loss, label='Training')
                    ax.plot(artifact.valid_history_trace, artifact.valid_history_loss, label='Validation')
                    ax.legend()
                    plt.xlabel('Training traces')
                    plt.ylabel('Loss')
                    plt.grid()
                    fig.tight_layout()
                    plot.add_plot(width=NoEscape(r'\textwidth'))
                    plot.add_caption('Loss plot.')

            doc.append(NoEscape(r'\newpage'))
            with doc.create(Section('Traces')):
                with doc.create(Tabularx('ll')) as table:
                    table.add_row(('Total training traces', '{:,}'.format(artifact.total_traces)))
                with doc.create(Subsection('Distributions encountered')):
                    with doc.create(Tabularx('ll')) as table:
                        num_distributions = len(artifact.one_hot_distribution.keys())
                        table.add_row(('Number of distributions', num_distributions))
                        table.add_empty_row()
                        for distribution in artifact.one_hot_distribution.keys():
                            table.add_row((distribution, ''))
                with doc.create(Subsection('Unique addresses encountered')):
                    with doc.create(Tabularx('lX')) as table:
                        num_addresses = len(artifact.one_hot_address.keys())
                        table.add_row(('Number of addresses', num_addresses))
                        address_collisions = max(0, num_addresses - artifact.one_hot_address_dim)
                        table.add_row(('Address collisions', address_collisions))
                        table.add_empty_row()
                    doc.append('\n')
                    with doc.create(LongTable('llp{16cm}')) as table:
                        # table.add_empty_row()
                        table.add_row('Count', 'ID', 'Unique address')
                        table.add_hline()

                        address_to_abbrev = {}
                        abbrev_to_address = {}
                        abbrev_i = 0
                        sorted_addresses = sorted(artifact.address_histogram.items(), key=lambda x:x[1], reverse=True)
                        plt_addresses = []
                        plt_counts = []
                        address_to_count = {}
                        address_count_total = 0
                        for address, count in sorted_addresses:
                            abbrev_i += 1
                            abbrev = 'A' + str(abbrev_i)
                            address_to_abbrev[address] = abbrev
                            abbrev_to_address[abbrev] = address
                            plt_addresses.append(abbrev)
                            plt_counts.append(count)
                            address_to_count[abbrev] = count
                            address_count_total += count
                            table.add_row(('{:,}'.format(count), abbrev, FootnoteText(address)))

                    with doc.create(Figure(position='H')) as plot:
                        fig = plt.figure(figsize=(10,5))
                        ax = plt.subplot(111)
                        plt_x = range(len(plt_addresses))
                        ax.bar(plt_x, plt_counts)
                        plt.xticks(plt_x, plt_addresses)
                        plt.xlabel('Unique address ID')
                        plt.ylabel('Count')
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Histogram of address hits.')

                with doc.create(Subsection('Lengths')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Min trace length', '{:,}'.format(artifact.trace_length_min)))
                        table.add_row(('Max trace length', '{:,}'.format(artifact.trace_length_max)))
                        s = 0
                        total_count = 0
                        for trace_length in artifact.trace_length_histogram:
                            count = artifact.trace_length_histogram[trace_length]
                            s += trace_length * count
                            total_count += count
                        trace_length_mean = s / total_count
                        table.add_row(('Mean trace length', '{:.2f}'.format(trace_length_mean)))
                    with doc.create(Figure(position='H')) as plot:
                        plt_lengths = [i for i in range(0, artifact.trace_length_max + 1)]
                        plt_counts = [artifact.trace_length_histogram[i] if i in artifact.trace_length_histogram else 0 for i in range(0, artifact.trace_length_max + 1)]
                        fig = plt.figure(figsize=(10,5))
                        ax = plt.subplot(111)
                        ax.bar(plt_lengths, plt_counts)
                        plt.xlabel('Length')
                        plt.ylabel('Count')
                        # plt.yscale('log')
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Histogram of trace lengths (of all traces used during training).')

                with doc.create(Subsection('Unique traces encountered')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Unique traces encountered', '{:,}'.format(len(artifact.trace_examples_histogram))))
                        table.add_row(('Unique trace memory capacity', '{:,}'.format(artifact.trace_examples_limit)))
                        table.add_row(('Unique traces rendered in detail', '{:,}'.format(min(len(artifact.trace_examples_histogram), opt.maxTraces))))
                    doc.append('\n')
                    with doc.create(LongTable('lllp{16cm}')) as table:
                        # table.add_empty_row()
                        table.add_row('Count', 'ID', 'Len.', 'Unique trace')
                        table.add_hline()

                        trace_to_abbrev = {}
                        abbrev_to_trace = {}
                        abbrev_to_addresses = {}
                        abbrev_i = 0
                        sorted_traces = sorted(artifact.trace_examples_histogram.items(), key=lambda x:x[1], reverse=True)
                        plt_traces = []
                        plt_counts = []
                        trace_to_count = {}
                        trace_count_total = 0
                        for trace, count in sorted_traces:
                            abbrev_i += 1
                            abbrev = 'T' + str(abbrev_i)
                            trace_to_abbrev[trace] = abbrev
                            abbrev_to_trace[abbrev] = trace
                            abbrev_to_addresses[abbrev] = list(map(lambda x:address_to_abbrev[x], artifact.trace_examples_addresses[trace]))
                            trace_addresses = abbrev_to_addresses[abbrev]
                            trace_addresses_repetitions = util.pack_repetitions(trace_addresses)
                            plt_traces.append(abbrev)
                            plt_counts.append(count)
                            trace_to_count[trace] = count
                            trace_count_total += count
                            length = len(artifact.trace_examples_addresses[trace])
                            table.add_row(('{:,}'.format(count), abbrev, '{:,}'.format(length), FootnoteText('-'.join([a + 'x' + str(i) if i>1 else a for a, i in trace_addresses_repetitions]))))

                    with doc.create(Figure(position='H')) as plot:
                        fig = plt.figure(figsize=(10,5))
                        ax = plt.subplot(111)
                        plt_x = range(len(plt_traces))
                        ax.bar(plt_x, plt_counts)
                        plt.xticks(plt_x, plt_traces)
                        plt.xlabel('Unique trace ID')
                        plt.ylabel('Count')
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Histogram of unique traces.')

                    with doc.create(Figure(position='H')) as plot:
                        master_trace_pairs = {}
                        transition_count_total = 0
                        for trace, count in sorted_traces:
                            ta = abbrev_to_addresses[trace_to_abbrev[trace]]
                            for left, right in zip(ta, ta[1:]):
                                if (left, right) in master_trace_pairs:
                                    master_trace_pairs[(left, right)] += count
                                else:
                                    master_trace_pairs[(left, right)] = count
                                transition_count_total += count
                        fig = plt.figure(figsize=(10,5))
                        ax = plt.subplot(111)
                        master_graph = pydotplus.graphviz.Dot(graph_type='digraph', rankdir='LR')
                        for p, w in master_trace_pairs.items():
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
                            master_graph.add_edge(pydotplus.Edge(n0,n1, weight=w))
                        for node in master_graph.get_nodes():
                            node.set_color('gray')
                            node.set_fontcolor('gray')
                        for edge in master_graph.get_edges():
                            edge.set_color('gray')


                        master_graph_annotated = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())
                        for node in master_graph_annotated.get_nodes():
                            color = util.rgb_to_hex(util.rgb_blend((1,1,1), (1,0,0), address_to_count[node.obj_dict['name']] / address_count_total))
                            node.set_style('filled')
                            node.set_fillcolor(color)
                            node.set_color('black')
                            node.set_fontcolor('black')
                        for edge in master_graph_annotated.get_edges():
                            (left, right) = edge.obj_dict['points']
                            count = master_trace_pairs[(left, right)]
                            edge.set_label(count)
                            color = util.rgb_to_hex((1.5*(count/transition_count_total),0,0))
                            edge.set_color(color)

                        png_str = master_graph_annotated.create_png(prog=['dot','-Gsize=15', '-Gdpi=600'])
                        bio = BytesIO()
                        bio.write(png_str)
                        bio.seek(0)
                        img = np.asarray(mpimg.imread(bio))
                        plt.imshow(util.crop_image(img), interpolation='bilinear')
                        plt.axis('off')
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Succession of unique address IDs (accumulated over all traces).')

                    for trace, count in sorted_traces[:opt.maxTraces]:
                        trace = trace_to_abbrev[trace]
                        doc.append(NoEscape(r'\newpage'))
                        with doc.create(Subsubsection('Unique trace ' + trace)):
                            sys.stdout.write('Rendering unique trace {0}...                                       \r'.format(trace))
                            sys.stdout.flush()

                            addresses = len(address_to_abbrev)
                            trace_addresses = abbrev_to_addresses[trace]

                            with doc.create(Tabularx('ll')) as table:
                                table.add_row(FootnoteText('Count'), FootnoteText('{:,}'.format(count)))
                                table.add_row(FootnoteText('Length'), FootnoteText('{:,}'.format(len(trace_addresses))))
                            doc.append('\n')

                            im = np.zeros((addresses, len(trace_addresses)))
                            for i in range(len(trace_addresses)):
                                address = trace_addresses[i]
                                address_i = plt_addresses.index(address)
                                im[address_i, i] = 1
                            truncate = 100
                            for col_start in range(0, len(trace_addresses), truncate):
                                col_end = min(col_start + truncate, len(trace_addresses))
                                with doc.create(Figure(position='H')) as plot:
                                    fig = plt.figure(figsize=(20 * ((col_end + 4 - col_start) / truncate),4))
                                    ax = plt.subplot(111)
                                    # ax.imshow(im,cmap=plt.get_cmap('Greys'))
                                    sns.heatmap(im[:,col_start:col_end], cbar=False, linecolor='lightgray', linewidths=.5, cmap='Greys',yticklabels=plt_addresses,xticklabels=np.arange(col_start,col_end))
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

                                fig = plt.figure(figsize=(10,5))
                                ax = plt.subplot(111)
                                graph = pydotplus.graphviz.graph_from_dot_data(master_graph.to_string())

                                trace_address_to_count = {}
                                for address in trace_addresses:
                                    if address in trace_address_to_count:
                                        trace_address_to_count[address] += 1
                                    else:
                                        trace_address_to_count[address] = 1


                                for p, w in pairs.items():
                                    left_node = graph.get_node(p[0])[0]
                                    right_node = graph.get_node(p[1])[0]
                                    edge = graph.get_edge(p[0], p[1])[0]

                                    color = util.rgb_to_hex(util.rgb_blend((1,1,1), (1,0,0), trace_address_to_count[p[0]] / len(trace_addresses)))
                                    left_node.set_style('filled')
                                    left_node.set_fillcolor(color)
                                    left_node.set_color('black')
                                    left_node.set_fontcolor('black')

                                    color = util.rgb_to_hex(util.rgb_blend((1,1,1), (1,0,0), trace_address_to_count[p[0]] / len(trace_addresses)))
                                    right_node.set_style('filled')
                                    right_node.set_fillcolor(color)
                                    right_node.set_color('black')
                                    right_node.set_fontcolor('black')

                                    (left, right) = edge.obj_dict['points']
                                    edge.set_label(w)
                                    color = util.rgb_to_hex((1.5*(w/len(trace_addresses)),0,0))
                                    edge.set_color(color)

                                png_str = graph.create_png(prog=['dot','-Gsize=30', '-Gdpi=600'])
                                bio = BytesIO()
                                bio.write(png_str)
                                bio.seek(0)
                                img = np.asarray(mpimg.imread(bio))
                                plt.imshow(util.crop_image(img), interpolation='bilinear')
                                plt.axis('off')
                                plot.add_plot(width=NoEscape(r'\textwidth'))
                                plot.add_caption('Succession of unique address IDs (for one trace of type ' + trace + ').')


                            with doc.create(Tabularx('lp{16cm}')) as table:
                                trace_addresses_repetitions = util.pack_repetitions(trace_addresses)
                                table.add_row(FootnoteText('Trace'), FootnoteText('-'.join([a + 'x' + str(i) if i>1 else a for a, i in trace_addresses_repetitions])))

            doc.generate_pdf(opt.saveReport, clean_tex=False)
            sys.stdout.write('                                                               \r')
            sys.stdout.flush()

    except KeyboardInterrupt:
        util.logger.log('Stopped')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
