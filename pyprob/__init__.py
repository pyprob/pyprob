__version__ = '0.10.0.dev8'

from pyprob.util import set_random_seed
from pyprob.util import set_cuda
from pyprob.util import get_config
from pyprob.model import Model, RemoteModel
from pyprob.state import sample, observe
