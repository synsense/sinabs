from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Optional, Union
from torch import nn
import torch
from sinabs.layers import SqueezeMixin, StatefulLayer


def _extract_single_input(input_: List[Any]) -> Any:
    if len(input_) != 1:
        raise ValueError("Multiple inputs not supported for `input_diff_hook`")
    return input_[0]

def conv_connection_map(
    layer: nn.Conv2d, input_shape: torch.Size, output_shape: torch.Size
) -> torch.Tensor:
    deconvolve = nn.ConvTranspose2d(
        layer.out_channels,
        layer.in_channels,
        kernel_size=layer.kernel_size,
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=False,
    )
    deconvolve.weight.data.fill_(1)
    deconvolve.weight.requires_grad = False
    # Set batch/time dimension to 1
    output_ones = torch.ones((1, *output_shape))
    connection_map = deconvolve(output_ones, output_size=(1 , *input_shape)).detach()
    connection_map.requires_grad = False
    connection_map = connection_map.to(layer.weight.device)
    return connection_map

def get_hook_data_dict(module):
    if not hasattr(module, "hook_data"):
        module.hook_data = dict()
    return module.hook_data

def input_diff_hook(module: Union[nn.Conv2d, nn.Linear], input_: List, output: Any):
    data = get_hook_data_dict(module)
    input_ = _extract_single_input(input_)

    # Difference between absolute output and output with absolute weights
    if isinstance(module, nn.Conv2d):
        abs_weight_output = torch.nn.functional.conv2d(
            input=input_,
            weight=torch.abs(module.weight),
            stride=module.stride,
            padding=module.padding,
            groups=module.groups,
        )
    else:
        abs_weight_output = nn.functional.linear(
            input=input_,
            weight=torch.abs(module.weight),
        )
    data["diff_output"] = abs_weight_output - torch.abs(output)


def firing_rate_hook(module: StatefulLayer, input_: Any, output: torch.Tensor):
    data = get_hook_data_dict(module)
    data["firing_rate"] = output.mean()

def firing_rate_per_neuron_hook(module: StatefulLayer, input_: Any, output: torch.Tensor):
    data = get_hook_data_dict(module)
    if isinstance(module, SqueezeMixin):
        # Common dimension for batch and time
        data["firing_rate_per_neuron"] = output.mean(0)
    else:
        # Output is of shape (N, T, ...)
        data["firing_rate_per_neuron"] = output.mean((0,1))

def conv_layer_synops_hook(module: nn.Conv2d, input_: List[torch.Tensor], output: torch.Tensor):
    data = get_hook_data_dict(module)
    input_ = _extract_single_input(input_)
    if (
        "connection_map" not in data
        # Ignore batch/time dimension when checking connectivity
        or data["connection_map"].shape[1:] != input_.shape[1:]
    ):
        new_connection_map = conv_connection_map(module, input_.shape[1:], output.shape[1:])
        data["connection_map"] = new_connection_map
    # Mean is across batches and timesteps
    data["layer_synops_per_timestep"] = (input_ * data["connection_map"]).mean(0).sum()

def linear_layer_synops_hook(module: nn.Linear, input_: List[torch.Tensor], output: torch.Tensor):
    data = get_hook_data_dict(module)
    input_ = _extract_single_input(input_)
    # Mean is across batches and timesteps
    synops = input_.mean(0).sum() * module.out_features
    data["layer_synops_per_timestep"] = synops


@dataclass
class ModelSynopsHook:
    dt: Optional[float] = None

    def __call__(self, module: nn.Sequential, input_: Any, output: Any):
        module_data = get_hook_data_dict(module)
        module_data["total_synops_per_timestep"] = 0.
        if self.dt is not None:
            module_data["total_synops_per_second"] = 0.
            
        scale_factors = []
        for lyr_idx, lyr in enumerate(module):
            if isinstance(lyr, nn.AvgPool2d):
                # Average pooling scales down the number of counted synops due to the averaging.
                # We need to correct for that by accumulating the scaling factors and multiplying
                # them to the counted Synops in the next conv or linear layer
                if lyr.kernel_size != lyr.stride:
                    warnings.warn(
                        f"In order for the Synops counter to work accurately the pooling "
                        f"layers kernel size should match their strides. At the moment at layer {name}, "
                        f"the kernel_size = {lyr.kernel_size}, the stride = {lyr.stride}."
                    )
                ks = lyr.kernel_size
                scaling = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
                scale_factors.append(scaling)
            if hasattr(lyr, "weight"):
                if (
                    hasattr(lyr, "hook_data")
                    and "layer_synops_per_timestep" in lyr.hook_data
                ):
                    data = lyr.hook_data
                    # Multiply all scale factors (or use 1 if empty)
                    scaling = reduce(lambda x, y: x*y, scale_factors, 1)
                    synops = data["layer_synops_per_timestep"] * scaling
                    data["synops_per_timestep"] = synops
                    module_data["total_synops_per_timestep"] += synops
                    if self.dt is not None:
                        synops_per_sec = data["synops_per_timestep"] / self.dt
                        data["synops_per_second"] = synops_per_sec
                        module_data["total_synops_per_second"] += synops_per_sec

                # For any module with weight: Reset `scale_factors` even if it doesn't count synops
                scale_factors = []

def register_synops_hooks(module: nn.Sequential, dt: Optional[float]=None):
    for lyr in module:
        if isinstance(lyr, nn.Conv2d):
            lyr.register_forward_hook(conv_layer_synops_hook)
        elif isinstance(lyr, nn.Linear):
            lyr.register_forward_hook(linear_layer_synops_hook)
    model_hook = ModelSynopsHook(dt)
    module.register_forward_hook(model_hook)
