from functools import partial
from typing import Optional, Tuple, Union

import nir
import nirtorch
import numpy as np
import torch
from torch import nn

import sinabs.layers as sl


def _as_pair(x) -> Tuple[int, int]:
    try:
        if len(x) == 1:
            return (x[0], x[0])
        elif len(x) >= 2:
            return tuple(x)
        else:
            raise ValueError()
    except TypeError:
        return x, x


def _import_sinabs_module(
    node: nir.NIRNode, batch_size: int, num_timesteps: int
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        linear = nn.Linear(
            in_features=node.weight.shape[1],
            out_features=node.weight.shape[0],
            bias=True,
        )
        linear.weight.data = node.weight
        linear.bias.data = node.bias
        return linear

    elif isinstance(node, nir.Conv1d):
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

    elif isinstance(node, nir.Conv2d):
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

    elif isinstance(node, nir.LI):
        if node.v_leak.shape == torch.Size([]):
            node.v_leak = node.v_leak.unsqueeze(0)
        if node.r.shape == torch.Size([]):
            node.r = node.r.unsqueeze(0)
        if any(node.v_leak != 0):
            raise ValueError("`v_leak` must be 0")
        if any(node.r != 1):
            raise ValueError("`r` must be 1")
        # TODO check for norm_input
        return sl.ExpLeakSqueeze(
            tau_mem=node.tau,
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            norm_input=False,
        )

    elif isinstance(node, nir.IF):
        return sl.IAFSqueeze(
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            spike_threshold=node.v_threshold,
        )

    elif isinstance(node, nir.LIF):
        if node.v_leak.shape == torch.Size([]):
            node.v_leak = node.v_leak.unsqueeze(0)
        if any(node.v_leak) != 0:
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
    elif isinstance(node, nir.SumPool2d):
        return sl.SumPool2d(kernel_size=node.kernel_size, stride=node.stride)
    elif isinstance(node, nir.Flatten):
        return nn.Flatten(start_dim=node.start_dim, end_dim=node.end_dim)


def from_nir(
    node: nir.NIRNode, batch_size: int = None, num_timesteps: int = None
) -> torch.nn.Module:
    return nirtorch.load(
        node,
        partial(
            _import_sinabs_module, batch_size=batch_size, num_timesteps=num_timesteps
        ),
    )


def _extend_to_shape(x: Union[torch.Tensor, float], shape: Tuple) -> torch.Tensor:
    if x.shape == shape:
        return x
    elif x.shape == (1,) or x.dim() == 0:
        return torch.ones(*shape) * x
    else:
        raise ValueError(f"Not sure how to extend {x} to shape {shape}")


def _extract_sinabs_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if type(module) in [sl.IAF, sl.IAFSqueeze]:
        return nir.IF(
            r=torch.ones_like(module.v_mem.detach()),
            v_threshold=_extend_to_shape(
                module.spike_threshold.detach(), module.v_mem.shape
            ),
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
            bias=module.bias.detach()
            if module.bias
            else torch.zeros((module.weight.shape[0])),
        )
    elif isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias.detach()
            if module.bias
            else torch.zeros((module.weight.shape[0])),
        )
    elif isinstance(module, sl.SumPool2d):
        return nir.SumPool2d(
            kernel_size=_as_pair(module.kernel_size),  # (Height, Width)
            stride=_as_pair(module.stride),  # (Height, width)
            padding=(0, 0),  # (Height, width)
        )
    elif isinstance(module, nn.Flatten):
        return nir.Flatten(
            input_type={"input": np.array([0, 0, 0, 0])},
            start_dim=module.start_dim,
            end_dim=module.end_dim,
        )
    raise NotImplementedError(f"Module {type(module)} not supported")


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "model"
) -> nir.NIRNode:
    return nirtorch.extract_nir_graph(
        module, _extract_sinabs_module, sample_data, model_name=model_name
    )
