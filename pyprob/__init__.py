from .util import __version__

from .util import set_random_seed
from .util import set_cuda
from .model import Model, ModelRemote
from .state import sample, observe

del util
del model
del state

# import time
# set_random_seed(int(time.time()))
# set_cuda(True)
