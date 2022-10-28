from pbr.version import VersionInfo

__version__ = VersionInfo("sinabs").release_string()

from .network import Network
from .synopcounter import SynOpCounter, SNNSynOpCounter
from .from_torch import from_model
from . import conversion, utils
