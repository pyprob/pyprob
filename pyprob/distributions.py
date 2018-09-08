import torch


class Distribution(object):
    def __init__(self, name, address_suffix='', torch_dist=None):
        self.name = name
        self.address_suffix = address_suffix
        self._torch_dist = torch_dist

    def sample(self):
        if self._torch_dist is not None:
            s = self._torch_dist.sample()
            return s
        else:
            raise NotImplementedError()
