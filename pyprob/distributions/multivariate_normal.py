import torch

from . import Distribution
from .. import util


class MultivariateNormal(Distribution):
    def __init__(self,
                 loc,
                 covariance_matrix=None,
                 precision_matrix=None,
                 scale_tril=None,
                 validate_args=None):
        loc = util.to_tensor(loc)
        scale = util.to_tensor(scale)
        covariance_matrix = util.to_tensor(covariance_matrix)
        precision_matrix = util.to_tensor(precision_matrix)
        scale_tril = util.to_tensor(scale_tril)
        super().__init__(name='Normal', address_suffix='Normal',
                         torch_dist=torch.distributions.MultivariateNormal(loc,
                                                                           covariance_matrix,
                                                                           precision_matrix,
                                                                           scale_tril,
                                                                           validate_args))

    def __repr__(self):
        return 'MultivariateNormal(mean:{}, ' \
               'covariance_matrix:{},' \
               'precision_matrix:{},' \
               'scale_tril:{})'.format(self.locs,
                                      self.covariance_matrix,
                                      self.precision_matrix,
                                      self.scale_tril)

    def cdf(self, value):
        return self._torch_dist.cdf(value)

    def icdf(self, value):
        return self._torch_dist.icdf(value)
