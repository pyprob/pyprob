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
        self._rows = []
        self._ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')

        self._logger = logging.getLogger()
        logger_file_handler = logging.FileHandler(file_name)
        logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self._logger.addHandler(logger_file_handler)
        self._logger.setLevel(logging.INFO)

        # self.log(colored('[] pyprob ' + pyprob.__version__, 'blue', attrs=['bold']))
        # self.log()
        # self.log('Started ' + util.get_time_str())
        # self.log()
        # self.log('Running on PyTorch ' + torch.__version__)
        # self.log()
        # self.log(colored('[] Hardware', 'blue', attrs=['bold']))
        # self.log()
        # if util.cuda_enabled:
        #     self.log('Running on    : CUDA')
        # else:
        #     self.log('Running on    : CPU')
        # cpu_info = cpuinfo.get_cpu_info()
        # if 'brand' in cpu_info:
        #     self.log('CPU           : {0}'.format(cpu_info['brand']))
        # else:
        #     self.log('CPU           : unknown')
        # if 'count' in cpu_info:
        #     self.log('CPU count     : {0} (logical)'.format(cpu_info['count']))
        # else:
        #     self.log('CPU count     : unknown')
        # if torch.cuda.is_available():
        #     self.log('CUDA          : available')
        #     self.log('CUDA devices  : {0}'.format(torch.cuda.device_count()))
        #     if util.cuda_enabled:
        #         if util.cuda_device == -1:
        #             self.log('CUDA selected : all')
        #         else:
        #             self.log('CUDA selected : {0}'.format(util.cuda_device))
        # else:
        #     self.log('CUDA          : not available')

    def remove_non_ascii(self, s):
        s = self._ansi_escape.sub('', s)
        return ''.join(i for i in s if ord(i)<128)

    def update(self):
        if self._in_jupyter:
            clear_output(wait=True)
            for row in self._rows:
                print(row)

    def reset(self):
        self._rows = []

    def log(self, line=''):
        self._rows.append(line)
        self._logger.info(self.remove_non_ascii(line))

    def log_error(self, line=''):
        line = colored('Error: ' + line, 'red', attrs=['bold'])
        self._rows.append(line)
        self._logger.error(self.remove_non_ascii(line))

    def log_warning(self, line=''):
        line = colored('Warning: ' + line, 'red', attrs=['bold'])
        self._rows.append(line)
        self._logger.warning(self.remove_non_ascii(line))
