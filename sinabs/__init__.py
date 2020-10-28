from pbr.version import VersionInfo

__version__ = VersionInfo('sinabs').release_string()

from .network import Network
from .synopcounter import SynOpCounter

from .utils import (
    get_activations,
    get_network_activations,
)
