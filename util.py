#
# Oxford Inference Compilation
# Utility functions
#
# Tuan-Anh Le, Atilim Gunes Baydin
# University of Oxford
# May 2016 -- March 2017
#

import time
import datetime
import logging
import sys
import re

version = '0.9.1'
epsilon = 1e-5

def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')

def init_logger(file_name):
    global logger
    logger = logging.getLogger()
    logger_file_handler = logging.FileHandler(file_name)
    logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logger_file_handler)
    logger.setLevel(logging.INFO)

def print_log(line):
    print(line)
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    logger.info(ansi_escape.sub('', line))
