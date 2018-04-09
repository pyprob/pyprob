import os
import sys
import datetime
import torch
from pylatex import Document, Section, Subsection, Subsubsection, Command, Figure, SubFigure, Tabularx, LongTable, FootnoteText, FlushLeft
from pylatex.utils import italic, NoEscape
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
import re


from . import util, __version__


def save_report(model, file_name):
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

    with doc.create(Section('Current system', numbering=False)):
        with doc.create(Tabularx('ll')) as table:
            table.add_row(('pyprob version', __version__))
            table.add_row(('PyTorch version', torch.__version__))

    # doc.append(NoEscape(r'\newpage'))
    with doc.create(Section('Inference network', numbering=False)):
        with doc.create(Section('File')):
            with doc.create(Tabularx('ll')) as table:
                # table.add_row(('File name', file_name))
                # file_size = '{:,}'.format(os.path.getsize(file_name))
                # table.add_row(('File size', file_size + ' Bytes'))
                table.add_row(('Created', inference_network._created))
                table.add_row(('Modified', inference_network._modified))
                table.add_row(('Updates to file', inference_network._updates))
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
            ax.plot(inference_network._history_train_loss_trace, inference_network._history_train_loss, label='Training')
            ax.plot(inference_network._history_valid_loss_trace, inference_network._history_valid_loss, label='Validation')
            ax.legend()
            plt.xlabel('Training traces')
            plt.ylabel('Loss')
            plt.grid()
            fig.tight_layout()
            plot.add_plot(width=NoEscape(r'\textwidth'))
            plot.add_caption('Loss plot.')

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
            with doc.create(Subsection('All modules')):
                doc.append(str(inference_network))

            for m_name, m in inference_network.named_modules():
                if (m_name != ''):
                    regex = r'(sample_embedding_layer\(\S*\)._)|(proposal_layer\(\S*\)._)|(_lstm)'
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
            doc.append(NoEscape(r'\newpage'))
            with doc.create(Subsection('Address embeddings')):
                for p_name, p in inference_network.named_parameters():
                    if ('address_embedding' in p_name):
                        with doc.create(Figure(position='H')) as plot:
                            fig = plt.figure(figsize=(10,10))
                            ax = plt.subplot(111)
                            plt.imshow(np.transpose(util.weights_to_image(p),(1,2,0)), interpolation='none')
                            plt.axis('off')
                            plot.add_plot(width=NoEscape(r'\textwidth'))
                            plot.add_caption(p_name)

    doc.generate_pdf(file_name, clean_tex=False)
    sys.stdout.write('                                                               \r')
    sys.stdout.flush()
