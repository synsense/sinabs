from pbr.version import VersionInfo

__version__ = VersionInfo("sinabs").release_string()

from . import conversion, utils
from .from_torch import from_model
from .network import Network
from .synopcounter import SNNAnalyzer, SynOpCounter
from .utils import reset_states, zero_grad
