from os import PathLike
from typing import Optional, Union

import nir
import torch
from nirtorch import extract_nir_graph
from torch import nn

import sinabs.layers as sl


def iaf_from_nir(
    node: nir.IF, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
):
    return sl.IAFSqueeze(
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        spike_threshold=node.v_threshold,
    )


def expleak_from_nir(
    node: nir.LI, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
):
    if node.v_leak != 0:
        raise ValueError("`v_leak` must be 0")

    if node.r != 1:
        raise ValueError("`r` must be 1")

    # TODO check for norm_input

    return sl.ExpLeakSqueeze(
        tau_mem=node.tau,
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        norm_input=False,
    )


def lif_from_nir(
    node: nir.LIF, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
):
    if node.v_leak != 0:
        raise ValueError("`v_leak` must be 0")

    # TODO check for norm_input

    return sl.LIFSqueeze(
        tau_mem=node.tau,
        min_v_mem=None,
        num_timesteps=num_timesteps,
        batch_size=batch_size,
        spike_threshold=node.v_threshold,
        tau_syn=None,
        norm_input=False,
    )


def linear_from_nir(
    node: nir.LIF, batch_size: Optional[int] = None, num_timesteps: Optional[int] = None
):
    linear = nn.Linear(
        in_features=node.weight.shape[1],
        out_features=node.weight.shape[0],
        bias=True,
    )
    linear.weight.data = node.weight
    linear.bias.data = node.bias
    return linear


def conv1d_from_nir(
    node: nir.Conv1d,
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
):
    conv = nn.Conv1d(
        in_channels=node.weight.shape[1],
        out_channels=node.weight.shape[0],
        kernel_size=node.weight.shape[2:],
        stride=node.stride,
        padding=node.padding,
        dilation=node.dilation,
        groups=node.groups,
        bias=True,
    )
    conv.weight.data = node.weight.float()
    conv.bias.data = node.bias.float()
    return conv


def conv2d_from_nir(
    node: nir.Conv2d,
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
):
    conv = nn.Conv2d(
        in_channels=node.weight.shape[1],
        out_channels=node.weight.shape[0],
        kernel_size=node.weight.shape[2:],
        stride=node.stride,
        padding=node.padding,
        dilation=node.dilation,
        groups=node.groups,
        bias=True,
    )
    conv.weight.data = node.weight.float()
    conv.bias.data = node.bias.float()
    return conv


node_conversion_functions = {
    nir.IF: iaf_from_nir,
    nir.LI: expleak_from_nir,
    nir.LIF: lif_from_nir,
    nir.Affine: linear_from_nir,
    nir.Conv1d: conv1d_from_nir,
    nir.Conv2d: conv2d_from_nir,
}


def from_nir(
    source: Union[PathLike, nir.NIRNode],
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
):
    """Generates a sinabs model from a NIR representation.

    Parameters:
        source: If string or Path will try to load from file. Otherwise should be NIR object
    """

    layers = [
        node_conversion_functions[type(node)](node, batch_size, num_timesteps)
        for node in source.nodes
        if type(node) != nir.Input and type(node) != nir.Output
    ]

    edge_array = torch.tensor(source.edges)
    # subtract source edges and check if all edges are sequential
    edge_array = edge_array - edge_array[:, :1]
    if edge_array[:, 0].sum() == 0 and edge_array[:, 1].sum() == edge_array.shape[0]:
        return nn.Sequential(*layers)
    else:
        raise NotImplementedError("Only sequential models are supported at the moment")


def _extract_sinabs_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if type(module) in [sl.IAF, sl.IAFSqueeze]:
        return nir.IF(
            r=torch.ones_like(module.v_mem.detach()),
            v_threshold=module.spike_threshold.detach(),
        )
    elif type(module) in [sl.LIF, sl.LIFSqueeze]:
        return nir.LIF(
            tau=module.tau_mem.detach(),
            v_threshold=module.spike_threshold.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    elif type(module) in [sl.ExpLeak, sl.ExpLeakSqueeze]:
        return nir.LI(
            tau=module.tau_mem.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())
    elif isinstance(module, torch.nn.Conv1d):
        return nir.Conv1d(
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias.detach(),
        )
    elif isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias.detach(),
        )
    raise NotImplementedError(f"Module {type(module)} not supported")


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "model"
) -> nir.NIRNode:
    return extract_nir_graph(
        module, _extract_sinabs_module, sample_data, model_name=model_name
    )
