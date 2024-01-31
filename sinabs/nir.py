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
        linear.weight.data = torch.tensor(node.weight).float()
        linear.bias.data = torch.tensor(node.bias).float()
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
        conv.weight.data = torch.tensor(node.weight).float()
        conv.bias.data = torch.tensor(node.bias).float()
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
        conv.weight.data = torch.tensor(node.weight).float()
        conv.bias.data = torch.tensor(node.bias).float()
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
            min_v_mem=-node.v_threshold,
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
        return sl.SumPool2d(
            kernel_size=tuple(node.kernel_size), stride=tuple(node.stride)
        )
    elif isinstance(node, nir.Flatten):
        start_dim = node.start_dim + 1 if node.start_dim >= 0 else node.start_dim
        end_dim = node.end_dim + 1 if node.end_dim >= 0 else node.end_dim
        return nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    elif isinstance(node, nir.Input):
        return nn.Identity()
    elif isinstance(node, nir.Output):
        return nn.Identity()


def from_nir(
    node: nir.NIRNode, batch_size: int = None, num_timesteps: int = None
) -> torch.nn.Module:
    """Load a sinabs model from an NIR model.

    Args:
        node (nir.NIRNode): An NIR node/graph of the model
        batch_size (int, optional): batch size of the data that is expected to be fed to the model.Defaults to None.
        num_timesteps (int, optional): Number of time steps per data sample. Defaults to None.

    NOTE:
        `batch_size` or `num_timesteps` has to be specified for the sinabs model to be instantiated correctly.

    Returns:
        torch.nn.Module: Returns a sinabs model that is equivalent to the NIR graph specified.
    """
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
        layer_shape = module.v_mem.shape[1:]
        nir_node = nir.IF(
            r=torch.ones(*layer_shape),  # Discard batch dim
            v_threshold=_extend_to_shape(module.spike_threshold.detach(), layer_shape),
        )
        return nir_node
    elif type(module) in [sl.LIF, sl.LIFSqueeze]:
        layer_shape = module.v_mem.shape[0]
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
            bias=(
                module.bias.detach()
                if module.bias
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=(
                module.bias.detach()
                if isinstance(module.bias, torch.Tensor)
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, sl.SumPool2d):
        return nir.SumPool2d(
            kernel_size=_as_pair(module.kernel_size),  # (Height, Width)
            stride=_as_pair(
                module.kernel_size if module.stride is None else module.stride
            ),  # (Height, width)
            padding=(0, 0),  # (Height, width)
        )
    elif isinstance(module, nn.Flatten):
        # Getting rid of the batch dimension for NIR
        start_dim = module.start_dim - 1 if module.start_dim > 0 else module.start_dim
        end_dim = module.end_dim - 1 if module.end_dim > 0 else module.end_dim
        return nir.Flatten(
            input_type=None,
            start_dim=start_dim,
            end_dim=end_dim,
        )
    raise NotImplementedError(f"Module {type(module)} not supported")


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "model"
) -> nir.NIRNode:
    """Generate a NIRGraph given a sinabs model.

    Args:
        module (torch.nn.Module): The sinabs model to be converted to NIR graph
        sample_data (torch.Tensor): A sample data that can be used to extract various shapes and internal states.
        model_name (str, optional): The name of the top level model. Defaults to "model".

    Returns:
        nir.NIRNode: Returns the equivalent NIR object.
    """
    return nirtorch.extract_nir_graph(
        module,
        _extract_sinabs_module,
        sample_data,
        model_name=model_name,
        ignore_dims=[0],
    )
