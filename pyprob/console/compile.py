#
# pyprob
# PyTorch-based library for probabilistic programming and inference compilation
# https://github.com/probprog/pyprob
#

import pyprob
from pyprob import util, logger
import traceback
import argparse
import sys

util.logger = logger.Logger('{0}/{1}'.format('.', 'pyprob-log' + util.get_time_stamp()))

def main():
    try:
        parser = argparse.ArgumentParser(description='pyprob ' + pyprob.__version__ + ' (Compilation Mode)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-v', '--version', help='show version information', action='store_true')
        parser.add_argument('--dir', help='directory for saving artifacts and logs', default='.')
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=-1, type=int)
        parser.add_argument('--parallel', help='parallelize on CUDA using DataParallel', action='store_true')
        parser.add_argument('--seed', help='random seed', default=123, type=int)
        parser.add_argument('--server', help='address of the probprog model server', default='tcp://127.0.0.1:5555')
        parser.add_argument('--optimizer', help='optimizer for training the artifact', choices=['adam', 'sgd'], default='adam', type=str)
        parser.add_argument('--learningRate', help='learning rate', default=0.0001, type=float)
        parser.add_argument('--momentum', help='momentum (only for sgd)', default=0.9, type=float)
        parser.add_argument('--weightDecay', help='L2 weight decay coefficient', default=0.0005, type=float)
        parser.add_argument('--clip', help='gradient clipping (-1: disabled)', default=-1, type=float)
        parser.add_argument('--batchSize', help='training batch size', default=64, type=int)
        parser.add_argument('--validSize', help='validation set size', default=256, type=int)
        parser.add_argument('--replaceValidBatch', help='replace the validation batch of a resumed artifact', action='store_true')
        parser.add_argument('--validInterval', help='validation interval (traces)', default=1000, type=int)
        parser.add_argument('--maxTraces', help='stop training after this many traces (-1: disabled)', default=-1, type=int)
        parser.add_argument('--oneHotDim', help='dimension for one-hot encodings', default=64, type=int)
        parser.add_argument('--standardize', help='standardize observations', action='store_true')
        parser.add_argument('--resume', help='resume training of the latest artifact', action='store_true')
        parser.add_argument('--obsReshape', help='reshape a 1d observation to a given shape (example: "1x10x10" will reshape 100 -> 1x10x10)', default=None, type=str)
        parser.add_argument('--obsEmb', help='observation embedding', choices=['fc', 'cnn1d2c', 'cnn2d6c', 'cnn3d4c', 'lstm'], default='fc', type=str)
        parser.add_argument('--obsEmbDim', help='observation embedding dimension', default=512, type=int)
        parser.add_argument('--smpEmbDim', help='sample embedding dimension', default=64, type=int)
        parser.add_argument('--lstmDim', help='lstm hidden unit dimension', default=512, type=int)
        parser.add_argument('--lstmDepth', help='number of stacked lstms', default=2, type=int)
        parser.add_argument('--dropout', help='dropout value', default=0.2, type=float)
        parser.add_argument('--softmaxBoost', help='multiplier before softmax', default=20.0, type=float)
        parser.add_argument('--keepArtifacts', help='keep all previously best artifacts during training, do not overwrite', action='store_true')
        parser.add_argument('--visdom', help='use Visdom for visualizations', action='store_true')
        parser.add_argument('--batchPool', help='use batches stored in files under the given path (instead of online training with ZMQ)', default='', type=str)
        parser.add_argument('--truncateBackprop', help='use truncated backpropagation through time if sequence length is greater than the given value (-1: disabled)', default=100, type=int)
        opt = parser.parse_args()

        if opt.version:
            print(pyprob.__version__)
            quit()

        if opt.batchPool == '':
            server = opt.server
            batch_pool = False
        else:
            server = opt.batchPool
            batch_pool = True

        util.set_random_seed(opt.seed)
        util.set_cuda(opt.cuda, opt.device)

        inference = pyprob.InferenceRemote(remote_server=server, batch_pool=batch_pool, standardize_observes=opt.standardize, directory=opt.dir, resume=opt.resume, lstm_dim=opt.lstmDim, lstm_depth=opt.lstmDepth, obs_emb=opt.obsEmb, obs_reshape=opt.obsReshape, obs_emb_dim=opt.obsEmbDim, smp_emb_dim=opt.smpEmbDim, one_hot_dim=opt.oneHotDim, softmax_boost=opt.softmaxBoost, dropout=opt.dropout, valid_size=opt.validSize)

        inference.compile(batch_size=opt.batchSize, valid_interval=opt.validInterval, optimizer_method=opt.optimizer, learning_rate=opt.learningRate, momentum=opt.momentum, weight_decay=opt.weightDecay, parallelize=opt.parallel, truncate_backprop=opt.truncateBackprop, grad_clip=opt.clip, max_traces=opt.maxTraces, keep_all_artifacts=opt.keepArtifacts, replace_valid_batch=opt.replaceValidBatch, valid_size=opt.validSize)


    except KeyboardInterrupt:
        util.log_print('Stopped')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
