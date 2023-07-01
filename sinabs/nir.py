from typing import Callable, Optional, Tuple, Type
from warnings import warn

import torch
from torch import nn

import sinabs
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential
from sinabs.conversion import replace_module


def from_nir_leaky(unit: nir.LeakyIntegrator):


def from_nir(
    source: Union[Path, str, NIR] 
):
    """
    Generates a sinabs model from a NIR representation

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert units to sinabs layers
