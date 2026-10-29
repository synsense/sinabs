import warnings
from functools import partial
from typing import Optional, Tuple, Union

import nir
import nirtorch
import numpy as np
import torch
from torch import nn

import sinabs.layers as sl
from sinabs.activation import MembraneReset, MembraneSubtract, SingleSpike


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


def _to_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _norm_input(tau: torch.Tensor, r: torch.Tensor, dt: float) -> bool:
    """Find the sinabs `norm_input` that matches the input resistance `r`.

    NIR describes the membrane as `tau * dv/dt = (v_leak - v) + r * I`. Sinabs
    adds the input directly (`v = alpha * v + I`) when `norm_input` is False,
    which corresponds to `r = tau / dt`, and scales it by `1 - alpha` when
    `norm_input` is True, which corresponds to `r = 1`.
    """
    if torch.allclose(r * dt / tau, torch.ones_like(r)):
        return False
    if torch.allclose(r, torch.ones_like(r)):
        return True
    raise ValueError(
        "Sinabs can only represent a resistance `r` of 1 or `tau / dt`, "
        f"but got r={r} with tau={tau} and dt={dt}."
    )


def _check_v_leak(node: nir.NIRNode):
    if np.any(np.asarray(node.v_leak) != 0):
        raise ValueError("`v_leak` must be 0")


def _reset_fn(node: nir.NIRNode) -> MembraneReset:
    v_reset = np.asarray(getattr(node, "v_reset", 0.0))
    if np.any(v_reset != v_reset.flat[0]):
        raise ValueError("Sinabs only supports a single `v_reset` per layer")
    return MembraneReset(reset_value=float(v_reset.flat[0]))


def _import_sinabs_module(
    node: nir.NIRNode, batch_size: int, num_timesteps: int, dt: float
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        linear = nn.Linear(
            in_features=node.weight.shape[1],
            out_features=node.weight.shape[0],
            bias=True,
        )
        linear.weight.data = _to_tensor(node.weight)
        linear.bias.data = _to_tensor(node.bias)
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
        conv.weight.data = _to_tensor(node.weight)
        conv.bias.data = _to_tensor(node.bias)
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
        conv.weight.data = _to_tensor(node.weight)
        conv.bias.data = _to_tensor(node.bias)
        return conv

    elif isinstance(node, nir.LI):
        _check_v_leak(node)
        tau = _to_tensor(node.tau)
        return sl.ExpLeakSqueeze(
            tau_mem=tau / dt,
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            norm_input=_norm_input(tau, _to_tensor(node.r), dt),
        )

    elif isinstance(node, nir.IF):
        r = _to_tensor(node.r)
        if not torch.allclose(r * dt, torch.ones_like(r)):
            raise ValueError(
                f"Sinabs can only represent a resistance `r` of 1 / dt, but got r={r} with dt={dt}."
            )
        v_threshold = _to_tensor(node.v_threshold)
        return sl.IAFSqueeze(
            min_v_mem=-v_threshold,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            spike_threshold=v_threshold,
            spike_fn=SingleSpike,
            reset_fn=_reset_fn(node),
        )

    elif isinstance(node, nir.LIF):
        _check_v_leak(node)
        tau = _to_tensor(node.tau)
        return sl.LIFSqueeze(
            tau_mem=tau / dt,
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            spike_threshold=_to_tensor(node.v_threshold),
            spike_fn=SingleSpike,
            reset_fn=_reset_fn(node),
            tau_syn=None,
            norm_input=_norm_input(tau, _to_tensor(node.r), dt),
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
    node: nir.NIRNode,
    batch_size: int = None,
    num_timesteps: int = None,
    dt: float = 1.0,
) -> torch.nn.Module:
    """Load a sinabs model from an NIR model.

    Args:
        node (nir.NIRNode): An NIR node/graph of the model
        batch_size (int, optional): batch size of the data that is expected to be fed to the model.Defaults to None.
        num_timesteps (int, optional): Number of time steps per data sample. Defaults to None.
        dt (float, optional): Duration of one sinabs time step, in the time unit of the NIR graph. Time constants are divided by `dt`. Graphs exported by snnTorch use `dt=1e-4`. Defaults to 1.0.

    NOTE:
        `batch_size` or `num_timesteps` has to be specified for the sinabs model to be instantiated correctly.
        NIR neurons emit at most one spike per time step and reset to `v_reset`, so spiking layers are created with `SingleSpike` and `MembraneReset`.

    Returns:
        torch.nn.Module: Returns a sinabs model that is equivalent to the NIR graph specified.
    """
    return nirtorch.load(
        node,
        partial(
            _import_sinabs_module,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            dt=dt,
        ),
    )


def _extend_to_shape(x: Union[torch.Tensor, float], shape: Tuple) -> torch.Tensor:
    if x.shape == shape:
        return x
    elif x.shape == (1,) or x.dim() == 0:
        return torch.ones(*shape) * x
    else:
        raise ValueError(f"Not sure how to extend {x} to shape {shape}")


def _export_reset(module: torch.nn.Module, layer_shape: Tuple) -> np.ndarray:
    if isinstance(module.reset_fn, MembraneReset):
        return np.full(layer_shape, module.reset_fn.reset_value, dtype=np.float32)
    if isinstance(module.reset_fn, MembraneSubtract):
        warnings.warn(
            f"{type(module).__name__} uses `MembraneSubtract`, which NIR cannot "
            "represent. It is exported as a reset to 0."
        )
    return np.zeros(layer_shape, dtype=np.float32)


def _extract_sinabs_module(
    module: torch.nn.Module, dt: float = 1.0
) -> Optional[nir.NIRNode]:
    if type(module) in [sl.IAF, sl.IAFSqueeze]:
        layer_shape = module.v_mem.shape[1:]  # Discard batch dim
        nir_node = nir.IF(
            r=np.full(layer_shape, 1.0 / dt, dtype=np.float32),
            v_threshold=_extend_to_shape(module.spike_threshold.detach(), layer_shape)
            .cpu()
            .numpy(),
            v_reset=_export_reset(module, layer_shape),
        )
        return nir_node
    elif type(module) in [sl.LIF, sl.LIFSqueeze]:
        layer_shape = module.v_mem.shape[1:]  # Discard batch dim
        tau_mem = _extend_to_shape(module.tau_mem.detach(), layer_shape).cpu()
        return nir.LIF(
            tau=tau_mem.numpy() * dt,
            v_threshold=_extend_to_shape(module.spike_threshold.detach(), layer_shape)
            .cpu()
            .numpy(),
            v_leak=np.zeros(layer_shape, dtype=np.float32),
            r=(
                np.ones(layer_shape, dtype=np.float32)
                if module.norm_input
                else tau_mem.numpy()
            ),
            v_reset=_export_reset(module, layer_shape),
        )
    elif type(module) in [sl.ExpLeak, sl.ExpLeakSqueeze]:
        layer_shape = module.v_mem.shape[1:]  # Discard batch dim
        tau_mem = _extend_to_shape(module.tau_mem.detach(), layer_shape).cpu()
        return nir.LI(
            tau=tau_mem.numpy() * dt,
            v_leak=np.zeros(layer_shape, dtype=np.float32),
            r=(
                np.ones(layer_shape, dtype=np.float32)
                if module.norm_input
                else tau_mem.numpy()
            ),
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
    module: torch.nn.Module,
    sample_data: torch.Tensor,
    model_name: str = "model",
    dt: float = 1.0,
) -> nir.NIRNode:
    """Generate a NIRGraph given a sinabs model.

    Args:
        module (torch.nn.Module): The sinabs model to be converted to NIR graph
        sample_data (torch.Tensor): A sample data that can be used to extract various shapes and internal states.
        model_name (str, optional): The name of the top level model. Defaults to "model".
        dt (float, optional): Duration of one sinabs time step, in the time unit of the NIR graph. Time constants are multiplied by `dt`. Use `dt=1e-4` for graphs that will be imported into snnTorch. Defaults to 1.0.

    Returns:
        nir.NIRNode: Returns the equivalent NIR object.
    """
    graph = nirtorch.extract_nir_graph(
        module,
        partial(_extract_sinabs_module, dt=dt),
        sample_data,
        model_name=model_name,
        ignore_dims=[0],
    )
    # Fill in the input and output shapes of nodes such as convolutions
    try:
        graph.infer_types()
    except ValueError as e:
        warnings.warn(f"Could not infer the types of the NIR graph: {e}")
    return graph
