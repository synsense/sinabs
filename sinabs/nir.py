from typing import Callable, Optional, Tuple, Type
from warnings import warn

import torch
from torch import nn

import sinabs
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential
from sinabs.conversion import replace_module


class ScalarFactor(nn.Module):
    def __init__(self, scale):
        super().__init__()
        # Make sure scale is a scalar
        self.scale = scale.item()
    def forward(self, x):
        return self.scale * x

def from_nir_leaky(unit: nir.LeakyIntegrator, batch_size: int):
    if v_leak != 0:
        raise ValueError("`v_leak` must be 0")
    
    scalar = ScalarFactor(unit.beta / unit.alpha)
   
   if unit.theta is None:
        leaky_element = from_nir_leaky_to_expleak(unit, batch_size=batch_size)
    else:
        leaky_element = from_nir_leaky_to_if(unit, batch_size=batch_size)

    return nn.Sequential(scalar, leaky_element)

def from_nir_leaky_to_expleak(unit: nir.LeakyIntegrator, batch_size: int):
    return sl.ExpLeakSqueeze(
        tau_mem = torch.from_numpy(unit.tau / unit.alpha),
        min_v_mem = None,
        norm_input = False,
    )

def from_nir_leaky_to_if(unit: nir.LeakyIntegrator, batch_size: int):
    return sl.LIFSqueeze(
        tau_mem = torch.from_numpy(unit.tau / unit.alpha),
        spike_threshold = unit.threshold,
        min_v_mem = None,
        norm_input = False,
        tau_syn = None,
    )

        

def from_nir(
    source: Union[Path, str, NIR], batch_size: int 
):
    """
    Generates a sinabs model from a NIR representation

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert units to sinabs layers
