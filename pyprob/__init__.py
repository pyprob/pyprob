__version__ = '0.10.0.dev11'

from .util import set_random_seed
from .util import set_cuda
from .model import Model
from .state import sample, observe

del util
del model
del state
