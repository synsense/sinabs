from typing import Optional, Union

import torch
from torch import nn
import sinabs.layers as sl


node_conversion_functions = {
    nir.LeakyIntegrator: expleak_from_nir,
    nir.LeakyIntegrateAndFire: lif_from_nir,
    nir.Linear: linear_from_nir,
    nir.Conv1d: conv1d_from_nir,
    nir.Conv2d: conv2d_from_nir,
}

def expleak_from_nir(
    node: nir.LI,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    if v_leak != 0:
        raise ValueError("`v_leak` must be 0")

    if node.alpha == node.tau:
        norm_input = False
    elif node.alpha == node.tau - 1:
        norm_input = True
    else:
        raise ValueError("`alpha` must be either `tau` or `tau-1`")

    return sl.LIFSqueeze(
        tau_mem=torch.from_numpy(node.tau),
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        tau_syn=None,
        norm_input=norm_input,
    )

def lif_from_nir(
    node: nir.LIF,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    if v_leak != 0:
        raise ValueError("`v_leak` must be 0")

    if node.alpha == node.tau:
        norm_input = False
    elif node.alpha == node.tau - 1:
        norm_input = True
    else:
        raise ValueError("`alpha` must be either `tau` or `tau-1`")

    return sl.LIFSqueeze(
        tau_mem=torch.from_numpy(node.tau),
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        spike_threshold=node.threshold,
        tau_syn=None,
        norm_input=norm_input,
    )


def linear_from_nir(
    node: nir.LIF,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    linear = nn.Linear(
        in_features=node.weights.shape[1],
        out_features=node.weights.shape[0],
        bias=True,
    )
    linear.weight.data = torch.from_numpy(node.weights).float()
    linear.bias.data = torch.from_numpy(node.bias).float()
    return linear


def conv1d_from_nir(
    node: nir.Conv1d,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    conv = nn.Conv1d(
        in_channels=node.weights.shape[1],
        out_channels=node.weights.shape[0],
        kernel_size=node.weights.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    )
    conv.weight.data = torch.from_numpy(node.weights).float()
    conv.bias.data = torch.from_numpy(node.bias).float()
    return conv


def conv2d_from_nir(
    node: nir.Conv2d,
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    conv = nn.Conv2d(
        in_channels=node.weights.shape[1],
        out_channels=node.weights.shape[0],
        kernel_size=node.weights.shape[2:],
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    )
    conv.weight.data = torch.from_numpy(node.weights).float()
    conv.bias.data = torch.from_numpy(node.bias).float()
    return conv


def from_nir(
    source: Union[Path, str, NIR], batch_size: int 
    batch_size: Optional[int]=None,
    num_timesteps: Optional[int]=None
):
    """
    Generates a sinabs model from a NIR representation

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert nodes to sinabs layers
    layers = [
        node_conversion_functions[type(node)](node)
        for node in source.nodes
    ]

    #TODO: Map NIRTorch graph to torch module
    
