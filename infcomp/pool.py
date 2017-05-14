#
# Oxford Inference Compilation
# https://arxiv.org/abs/1610.09900
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- May 2017
#

import infcomp
from infcomp import util
import sys
import io
import os
from termcolor import colored
import random

class Requester(object):
    def __init__(self, pool_path):
        self.pool_path = pool_path
        num_files = len(self.current_files())
        util.log_print(colored('Protocol: working with batch pool (currently with {0} files) at {1}'.format(num_files, pool_path), 'yellow', attrs=['bold']))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def current_files(self):
        files = [name for name in os.listdir(self.pool_path)]
        files = list(map(lambda f:os.path.join(self.pool_path, f), files))
        return files

    def close(self):
        num_files = len(self.current_files())
        util.log_print(colored('Protocol: leaving batch pool (currently with {0} files) at {1}'.format(num_files, pool_path), 'yellow', attrs=['bold']))

    def send_request(self, request):
        return

    def receive_reply(self):
        current_files = self.current_files()
        current_file = random.choice(current_files)
        with open(current_file, 'rb') as f:
            return bytearray(f.read())
