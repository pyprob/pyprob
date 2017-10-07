import pyprob
from pyprob import util
import torch
import logging
import re
import cpuinfo
from termcolor import colored
from IPython.display import clear_output

class Logger(object):
    def __init__(self, file_name):
        self._file_name = file_name
        self._in_jupyter = util.in_jupyter()
        self._jupyter_rows = []
        self._ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

        self._logger = logging.getLogger()
        logger_file_handler = logging.FileHandler(file_name)
        logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self._logger.addHandler(logger_file_handler)
        self._logger.setLevel(logging.INFO)

    def remove_non_ascii(self, s):
        s = self._ansi_escape.sub('', s)
        return ''.join(i for i in s if ord(i)<128)

    def _jupyter_update(self):
        if self._in_jupyter:
            clear_output(wait=True)
            for row in self._jupyter_rows:
                print(row)

    def reset(self):
        self._jupyter_rows = []

    def log(self, line=''):
        if self._in_jupyter:
            self._jupyter_rows.append(line)
            self._jupyter_update()
        else:
            print(line)
        self._logger.info(self.remove_non_ascii(line))

    def log_error(self, line=''):
        line = colored('Error: ' + line, 'red', attrs=['bold'])
        if self._in_jupyter:
            self._jupyter_rows.append(line)
            self.jupyter_update()
        else:
            print(line)
        self._logger.error(self.remove_non_ascii(line))

    def log_warning(self, line=''):
        line = colored('Warning: ' + line, 'red', attrs=['bold'])
        if self._in_jupyter:
            self._jupyter_rows.append(line)
            self._jupyter_update()
        else:
            print(line)
        self._logger.warning(self.remove_non_ascii(line))

    def log_config(self):
        line0 = util.get_config()
        if not self._in_jupyter:
            print()
            print(line0)
            print()
        self._logger.info('')
        self._logger.info(self.remove_non_ascii(line0))
        self._logger.info('')

    def log_compile_begin(self, server, time_str, time_improvement_str, trace_str, traces_per_sec_str):
        line1 = colored('Training from ' + server, 'blue', attrs=['bold'])
        line2 = '{{:{0}}}'.format(len(time_str)).format('Train. time') + ' │ ' + '{{:{0}}}'.format(len(trace_str)).format('Trace') + ' │ Training loss   │ Min.train.loss│ Valid. loss     |' + '{{:{0}}}'.format(len(time_improvement_str)).format('T.since best') + ' │ TPS'
        line3 = '─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼───────────────┼─────────────────┼─' + '─'*len(time_improvement_str) + '─┼─' + '─'*len(traces_per_sec_str)
        if not self._in_jupyter:
            print()
            print(line1)
            print()
            print(line2)
            print(line3)
        self._logger.info('')
        self._logger.info(self.remove_non_ascii(line1))
        self._logger.info('')
        self._logger.info(self.remove_non_ascii(line2))
        self._logger.info(self.remove_non_ascii(line3))

    def log_compile_valid(self, time_str, time_improvement_str, trace_str, traces_per_sec_str):
        line = '─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼───────────────┼─────────────────┼─' + '─'*len(time_improvement_str) + '─┼─' + '─'*len(traces_per_sec_str)
        if not self._in_jupyter:
            print(line)
        self._logger.info(self.remove_non_ascii(line))


    def log_compile(self, time_str, time_session_start_str, time_best_str, time_improvement_str, trace_str, trace_session_start_str, trace_best_str, train_loss_str, train_loss_session_start_str, train_loss_best_str, valid_loss_str, valid_loss_session_start_str, valid_loss_best_str, traces_per_sec_str):
        line = '{0} │ {1} │ {2} │ {3} │ {4} │ {5} │ {6}'.format(time_str, trace_str, train_loss_str, train_loss_best_str, valid_loss_str, time_improvement_str, traces_per_sec_str)
        if self._in_jupyter:
            self.reset()
            self._jupyter_rows.append('────────┬─' + '─'*len(time_str) + '─┬─' + '─'*len(trace_str) + '─┬─────────────────┬─────────────────')
            self._jupyter_rows.append('        │ {0:>{1}} │ {2:>{3}} │ Training loss   │ Valid. loss     '.format('Train. time', len(time_str), 'Trace', len(trace_str)))
            self._jupyter_rows.append('────────┼─' + '─'*len(time_str) + '─┼─' + '─'*len(trace_str) + '─┼─────────────────┼─────────────────')
            self._jupyter_rows.append('Start   │ {0:>{1}} │ {2:>{3}} │ {4}   │ {5}'.format(time_session_start_str, len(time_str), trace_session_start_str, len(trace_str), train_loss_session_start_str, valid_loss_session_start_str))
            self._jupyter_rows.append('Best    │ {0:>{1}} │ {2:>{3}} │ {4}   │ {5}'.format(time_best_str, len(time_str), trace_best_str, len(trace_str), train_loss_best_str, valid_loss_best_str))
            self._jupyter_rows.append('Current │ {0} │ {1} │ {2} │ {3}'.format(time_str, trace_str, train_loss_str, valid_loss_str))
            self._jupyter_rows.append('────────┴─' + '─'*len(time_str) + '─┴─' + '─'*len(trace_str) + '─┴─────────────────┴─────────────────')
            self._jupyter_rows.append('Training on {0}, {1} traces/s'.format('CUDA' if util.cuda_enabled else 'CPU', traces_per_sec_str))
            self._jupyter_update()
        else:
            print(line)
        self._logger.info(self.remove_non_ascii(line))
