import os
import torch
from ...distributions import Normal, Uniform, Categorical


class MyFile:
    """ Makes h5py support multiprocessing (needed for num_workers > 1 used with pyprob dataloader)

    See: https://github.com/h5py/h5py/issues/934

    """

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.pos = 0

    def seek(self, pos, whence=0):
        if whence == 0:
            self.pos = pos
        elif whence == 1:
            self.pos += pos
        else:
            self.pos = os.lseek(self.fd, pos, whence)
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size):
        b = os.pread(self.fd, size, self.pos)
        self.pos += len(b)
        return b


def construct_distribution(name, dist_args):

    if name == 'Normal':
        return Normal(**dist_args)
    elif name == 'Uniform':
        return Uniform(**dist_args)
    elif name == 'Categorical':
        return Categorical(**dist_args)
    else:
        raise NotImplementedError("Distribution not supported to save on disk")
