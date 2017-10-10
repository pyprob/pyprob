#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util
import traceback
import argparse
import sys

def main():
    try:
        parser = argparse.ArgumentParser(description='pyprob ' + pyprob.__version__ + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for saving artifacts and logs', default='.')
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--seed', help='random seed', default=123, type=int)
        parser.add_argument('--server', help='address of the probprog model server', default='tcp://0.0.0.0:6666')
        opt = parser.parse_args()

        if opt.version:
            print(pyprob.__version__)
            quit()

        util.set_random_seed(opt.seed)
        util.set_cuda(opt.cuda, opt.device)

        inference = pyprob.InferenceRemote(local_server=opt.server, directory=opt.dir, resume=True)
        inference.infer()

    except KeyboardInterrupt:
        util.log_print('Stopped')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)
