from pbr.version import VersionInfo

__version__ = VersionInfo("sinabs").release_string()

from . import conversion, utils, validate_memory_speck
from .from_torch import from_model
from .network import Network
from .nir import from_nir, to_nir
from .synopcounter import SNNAnalyzer, SynOpCounter
from .utils import (
    reset_states,
    set_batch_size,
    zero_grad,
    validate_memory_mapping_speck,
)
from .validate_memory_speck import ValidateMapping
