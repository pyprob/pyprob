__version__ = '0.10.0.dev2'

from pyprob.util import set_random_seed
from pyprob.util import set_cuda
from pyprob.util import get_config
from pyprob import inference
from pyprob.inference import InferenceRemote
from pyprob.state import sample, observe
