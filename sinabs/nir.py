from os import PathLike
from typing import Optional, Union

import nir
import torch
from nirtorch import extract_nir_graph
from torch import nn

import sinabs.layers as sl


def expleak_from_nir(
    node: nir.LI, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
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
    node: nir.LIF, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
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
    node: nir.LIF, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
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
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
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
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
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


node_conversion_functions = {
    nir.LI: expleak_from_nir,
    nir.LIF: lif_from_nir,
    nir.Linear: linear_from_nir,
    nir.Conv1d: conv1d_from_nir,
    nir.Conv2d: conv2d_from_nir,
}


def from_nir(
    source: Union[PathLike, nir.NIR],
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
):
    """Generates a sinabs model from a NIR representation.

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    # Convert nodes to sinabs layers
    layers = [node_conversion_functions[type(node)](node) for node in source.nodes]

    # TODO: Map NIRTorch graph to torch module


def _extract_sinabs_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if type(module) == sl.LIF:
        return nir.LIF(
            tau=module.tau_mem.detach(),
            v_threshold=module.spike_threshold.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    if type(module) == sl.ExpLeak:
        return nir.LI(
            tau=module.tau_mem.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Linear(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Linear(module.weight.detach(), module.bias.detach())

    return None


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "norse"
) -> nir.NIR:
    return extract_nir_graph(
        module, _extract_sinabs_module, sample_data, model_name=model_name
    )
