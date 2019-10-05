import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose

from . import EmbeddingFeedForward
from .. import util
from ..distributions import Distribution, Categorical

class SurrogateCategorical(nn.Module):
    # only support 1 d distributions
    def __init__(self, input_shape, num_categories, constants={}, num_layers=2,
                 hidden_dim=None):
        super().__init__()
        input_shape = util.to_size(input_shape)
        self._ff = EmbeddingFeedForward(input_shape=input_shape,
                                        output_shape=torch.Size([num_categories]),
                                        num_layers=num_layers,
                                        activation_last=None,
                                        hidden_dim=hidden_dim)
        self._total_train_iterations = 0
        self.num_categories = num_categories
        self._logsoftmax = nn.LogSoftmax(dim=1)

        self.dist_type = Categorical(probs=torch.ones([self.num_categories])/self.num_categories)

    def forward(self, x, no_batch=False):
        batch_size = x.size(0)
        x = self._ff(x)
        self._logits = self._logsoftmax(x).view(batch_size, self.num_categories)

        if no_batch:
            self._logits = self._logits.squeeze(0)

        return Categorical(logits=self._logits)

    def _loss(self, values):
        q_categorical = Categorical(self._logits)
        return -q_categorical.log_prob(values)

        #return Distribution.kl_divergence(p_categorical, q_categorical)
