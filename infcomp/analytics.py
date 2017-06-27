#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Atilim Gunes Baydin, Tuan Anh Le, Mario Lezcano Casado, Frank Wood
# University of Oxford
# May 2016 -- June 2017
#

import infcomp
from infcomp import util
import torch
import argparse
from termcolor import colored
import datetime
import sys
import os
from pprint import pformat
import matplotlib
matplotlib.use('Agg') # Do not use X server
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['axes.axisbelow'] = True
import seaborn as sns
sns.set_style("ticks")
flatui = ["#3b5b92", "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest
import csv
import traceback
from pylatex import Document, Section, Subsection, Subsubsection, Command, Figure, Tabularx, LongTable, FootnoteText
from pylatex.utils import italic, NoEscape

def main():
    try:
        parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Analytics)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for loading artifacts and saving logs', default='.')
        parser.add_argument('--nth', help='show the nth artifact (-1: last, -2: second-to-last, etc.)', type=int, default=-1)
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--structure', help='show extra information about artifact structure', action='store_true')
        parser.add_argument('--saveReport', help='save a full analytics report (tex and pdf)', type=str)
        parser.add_argument('--saveLoss', help='save training and validation loss history (csv)', type=str)
        parser.add_argument('--saveAddresses', help='save histogram of addresses (csv)', type=str)
        parser.add_argument('--saveTraceLengths', help='save histogram of trace lengths (csv)', type=str)
        parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
        opt = parser.parse_args()

        if opt.version:
            print(infcomp.__version__)
            quit()

        time_stamp = util.get_time_stamp()
        util.init_logger('{0}/{1}'.format(opt.dir, 'infcomp-analytics-log' + time_stamp))
        util.init(opt, 'Analytics')

        util.log_print()
        util.log_print(colored('[] Artifact', 'blue', attrs=['bold']))
        util.log_print()

        file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp-artifact'), opt.nth)
        artifact = util.load_artifact(file_name, opt.cuda, opt.device)

        if opt.structure:
            util.log_print()
            util.log_print(colored('[] Artifact structure', 'blue', attrs=['bold']))
            util.log_print()

            util.log_print(artifact.get_structure_str())
            util.log_print(artifact.get_parameter_str())

        if opt.saveLoss:
            util.log_print('Saving training and validation loss history to file: ' + opt.saveLoss)
            with open(opt.saveLoss, 'w') as f:
                data = [artifact.train_history_trace, artifact.train_history_loss, artifact.valid_history_trace, artifact.valid_history_loss]
                writer = csv.writer(f)
                writer.writerow(['train_trace', 'train_loss', 'valid_trace', 'valid_loss'])
                for values in zip_longest(*data):
                    writer.writerow(values)

        if opt.saveAddresses:
            util.log_print('Saving address histogram to file: ' + opt.saveAddresses)
            with open(opt.saveAddresses, 'w') as f:
                data_address = []
                data_count = []
                for address in artifact.address_histogram:
                    data_address.append(address)
                    data_count.append(artifact.address_histogram[address])
                data = [data_address, data_count]
                writer = csv.writer(f)
                writer.writerow(['address', 'count'])
                for values in zip_longest(*data):
                    writer.writerow(values)

        if opt.saveTraceLengths:
            util.log_print('Saving trace length histogram to file: ' + opt.saveTraceLengths)
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
            util.log_print('Saving analytics report to files: ' + opt.saveReport + '.tex and ' + opt.saveReport + '.pdf')

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
                    table.add_row(('InfComp version', infcomp.__version__))
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
                        table.add_row(('InfComp version', artifact.code_version))
                        table.add_row(('PyTorch version', artifact.pytorch_version))
                        table.add_row(('Trained on', 'CUDA' if artifact.on_cuda else 'CPU'))
                with doc.create(Subsection('Neural network')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Trainable parameters', '{:,}'.format(artifact.num_params_history_num_params[-1])))
                        table.add_row(('LSTM input size', artifact.lstm_input_dim))
                        table.add_row(('LSTM hidden units', artifact.lstm_dim))
                        table.add_row(('LSTM depth', artifact.lstm_depth))
                        table.add_row(('Softmax boost', artifact.softmax_boost))
                        table.add_row(('Dropout', artifact.dropout))
                        table.add_row(('Standardize inputs', artifact.standardize))
                    with doc.create(Figure(position='H')) as plot:
                        fig = plt.figure(figsize=(10,4))
                        ax = plt.subplot(111)
                        ax.plot(artifact.num_params_history_trace, artifact.num_params_history_num_params)
                        plt.xlabel('Traces')
                        plt.ylabel('Number of parameters')
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Number of parameters.')
                    with doc.create(Subsubsection('Observation embedding')):
                        with doc.create(Tabularx('ll')) as table:
                            table.add_row(('Embedding', artifact.obs_emb))
                            table.add_row(('Size', artifact.obs_emb_dim))
                    with doc.create(Subsubsection('Sample embeddings')):
                        with doc.create(Tabularx('ll')) as table:
                            table.add_row(('Embedding', artifact.smp_emb))
                            table.add_row(('Size', artifact.smp_emb_dim))
                    with doc.create(Subsubsection('PyTorch module structure')):
                        doc.append(artifact.get_structure_str())
                    with doc.create(Subsubsection('PyTorch parameter tensors')):
                        doc.append(artifact.get_parameter_str())

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
                    plt.xlabel('Traces')
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
                        for address, count in sorted_addresses:
                            abbrev_i += 1
                            abbrev = 'A' + str(abbrev_i)
                            address_to_abbrev[address] = abbrev
                            abbrev_to_address[abbrev] = address
                            plt_addresses.append(abbrev)
                            plt_counts.append(count)
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
                        plt.grid()
                        fig.tight_layout()
                        plot.add_plot(width=NoEscape(r'\textwidth'))
                        plot.add_caption('Histogram of trace lengths.')

                with doc.create(Subsection('Unique traces encountered')):
                    with doc.create(Tabularx('ll')) as table:
                        table.add_row(('Saved unique traces', '{:,}'.format(len(artifact.trace_examples_histogram))))
                        table.add_row(('Unique trace memory limit', '{:,}'.format(artifact.trace_examples_limit)))
                    doc.append('\n')
                    with doc.create(LongTable('lllp{16cm}')) as table:
                        # table.add_empty_row()
                        table.add_row('Count', 'ID', 'Length', 'Unique trace')
                        table.add_hline()

                        trace_to_abbrev = {}
                        abbrev_to_trace = {}
                        abbrev_to_addresses = {}
                        abbrev_i = 0
                        sorted_traces = sorted(artifact.trace_examples_histogram.items(), key=lambda x:x[1], reverse=True)
                        plt_traces = []
                        plt_counts = []
                        for trace, count in sorted_traces:
                            abbrev_i += 1
                            abbrev = 'T' + str(abbrev_i)
                            trace_to_abbrev[trace] = abbrev
                            abbrev_to_trace[abbrev] = trace
                            abbrev_to_addresses[abbrev] = list(map(lambda x:address_to_abbrev[x], artifact.trace_examples_addresses[trace]))
                            plt_traces.append(abbrev)
                            plt_counts.append(count)
                            length = len(artifact.trace_examples_addresses[trace])
                            table.add_row(('{:,}'.format(count), abbrev, '{:,}'.format(length), FootnoteText('-'.join(abbrev_to_addresses[abbrev]))))

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

                    for trace, _ in sorted_traces[:10]:
                        trace = trace_to_abbrev[trace]
                        doc.append(NoEscape(r'\newpage'))
                        with doc.create(Subsubsection('Unique trace ' + trace)):
                            addresses = len(address_to_abbrev)
                            trace_addresses = abbrev_to_addresses[trace]

                            im = np.zeros((addresses, len(trace_addresses)))
                            for i in range(len(trace_addresses)):
                                address = trace_addresses[i]
                                address_i = plt_addresses.index(address)
                                im[address_i, i] = 1
                            with doc.create(Figure(position='H')) as plot:
                                fig = plt.figure(figsize=(10,2))
                                ax = plt.subplot(111)
                                # ax.imshow(im,cmap=plt.get_cmap('Greys'))
                                sns.heatmap(im, cbar=False, linecolor='gray', linewidths=.5, cmap='Greys',yticklabels=plt_addresses)
                                plt.yticks(rotation=0)
                                fig.tight_layout()
                                plot.add_plot(width=NoEscape(r'\textwidth'))
                                plot.add_caption('Unique trace ' + trace + '.')
                            doc.append(FootnoteText('Full trace:\n'))
                            doc.append(FootnoteText('-'.join(trace_addresses) + '\n'))

                            doc.append(FootnoteText('Compact trace:\n'))
                            trace_addresses_repetitions = util.pack_repetitions(trace_addresses)
                            doc.append(FootnoteText('-'.join([a + 'x' + str(i) for a, i in trace_addresses_repetitions])))
                            doc.append('\n\n')

                            doc.append(FootnoteText('Sorted repetitions\n'))
                            sorted_repetitions = sorted(trace_addresses_repetitions, key=lambda x:x[1], reverse=True)
                            with doc.create(Tabularx('ll')) as table:
                                table.add_row(FootnoteText('Count'), FootnoteText('Unique address ID'))
                                table.add_hline()
                                for a, i in sorted_repetitions:
                                    table.add_row(FootnoteText(i), FootnoteText(a))
            doc.generate_pdf(opt.saveReport, clean_tex=False)

    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
