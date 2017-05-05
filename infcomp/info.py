#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
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
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest
import csv
import traceback

def main():
    try:
        parser = argparse.ArgumentParser(description='Oxford Inference Compilation ' + infcomp.__version__ + ' (Artifact Info)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory to save artifacts and logs', default='.')
        parser.add_argument('--nth', help='show the nth artifact (-1: last, -2: second-to-last, etc.)', type=int, default=-1)
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=4, type=int)
        parser.add_argument('--structure', help='show extra information about artifact structure', action='store_true')
        parser.add_argument('--savePlot', help='save loss plot to file (supported formats: eps, jpg, png, pdf, svg, tif)', type=str)
        parser.add_argument('--showPlot', help='show the loss plot in screen', action='store_true')
        parser.add_argument('--saveHist', help='save the training and validation loss history (csv)', type=str)
        parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
        opt = parser.parse_args()

        if opt.version:
            print(infcomp.__version__)
            quit()

        time_stamp = util.get_time_stamp()
        util.init_logger('{0}/{1}'.format(opt.dir, 'infcomp-info-log' + time_stamp))
        util.init(opt, 'Artifact Info')

        util.log_print()
        util.log_print(colored('[] Artifact', 'blue', attrs=['bold']))
        util.log_print()

        file_name = util.file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp-artifact'), opt.nth)
        artifact = util.load_artifact(file_name, opt.cuda, opt.device)

        if opt.structure:
            util.log_print()
            util.log_print(colored('[] Artifact structure', 'blue', attrs=['bold']))
            util.log_print()

            util.log_print(artifact.get_structure())

        if opt.showPlot or opt.savePlot:
            util.log_print()
            util.log_print(colored('[] Loss plot', 'blue', attrs=['bold']))
            util.log_print()
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(artifact.train_history_trace, artifact.train_history_loss, label='Training')
            ax.plot(artifact.valid_history_trace, artifact.valid_history_loss, label='Validation')
            ax.legend()
            plt.xlabel('Traces')
            plt.ylabel('Loss')
            plt.grid()
            fig.tight_layout()
            if opt.showPlot:
                util.log_print('Plotting loss to screen')
                plt.show()
            if not opt.savePlot is None:
                util.log_print('Saving loss plot to file: ' + opt.savePlot)
                fig.savefig(opt.savePlot)

        if not opt.saveHist is None:
            util.log_print('Saving training and validation loss history to file: ' + opt.saveHist)
            with open(opt.saveHist, 'w') as f:
                data = [artifact.train_history_trace, artifact.train_history_loss, artifact.valid_history_trace, artifact.valid_history_loss]
                writer = csv.writer(f)
                writer.writerow(['train_trace', 'train_loss', 'valid_trace', 'valid_loss'])
                for values in zip_longest(*data):
                    writer.writerow(values)

    except KeyboardInterrupt:
        util.log_print('Shutdown requested')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
