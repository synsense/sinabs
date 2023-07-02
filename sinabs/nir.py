from typing import Callable, Optional, Tuple, Type
from warnings import warn

import torch
from torch import nn

import sinabs
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential
from sinabs.conversion import replace_module


unit_conversion_functions = {
    nir.LeakyIntegrator: expleak_from_nir,
    nir.LeakyIntegrateAndFire: lif_from_nir,
    nir.Linear: linear_from_nir,
    nir.Conv1d: conv1d_from_nir,
    nir.Conv2d: conv2d_from_nir,
}

def from_nir_leaky(
    unit: nir.LeakyIntegrator,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    if v_leak != 0:
        raise ValueError("`v_leak` must be 0")

    parameters = dict(
        tau_mem=torch.from_numpy(unit.tau),
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
    )

    if unit.alpha == unit.tau:
        parameters["norm_input"] = False
    elif unit.alpha == unit.tau - 1:
        parameters["norm_input"] = True
    else:
        raise ValueError("`alpha` must be either `tau` or `tau-1`")

    if unit.theta is None:
        return sl.ExpLeakSqueeze(**parameters)
    else:
        return sl.LIFSqueeze(
            **parameters,
            spike_threshold=unit.threshold,
            tau_syn=None,
        )
    return nn.Sequential(scalar, leaky_element)

def from_nir(
    source: Union[Path, str, NIR], batch_size: int 
):
    """
    Generates a sinabs model from a NIR representation

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert units to sinabs layers
