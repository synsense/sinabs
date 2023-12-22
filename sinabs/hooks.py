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

def input_diff_hook(module: Union[nn.Conv2d, nn.Linear], input_: List, output: Any):
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
    module.diff_output = abs_weight_output - torch.abs(output)


def firing_rate_hook(module: StatefulLayer, input_: Any, output: torch.Tensor):
    module.firing_rate = output.mean()

def firing_rate_per_neuron_hook(module: StatefulLayer, input_: Any, output: torch.Tensor):
    if isinstance(module, SqueezeMixin):
        # Common dimension for batch and time
        module.firing_rate_per_neuron = output.mean(0)
    else:
        # Output is of shape (N, T, ...)
        module.firing_rate_per_neuron = output.mean((0,1))

def conv_layer_synops_hook(module: nn.Conv2d, input_: List[torch.Tensor], output: torch.Tensor):
    input_ = _extract_single_input(input_)
    if (
        not hasattr(module, "connection_map")
        # Ignore batch/time dimension when checking connectivity
        or module.connection_map.shape[1:] != input_.shape[1:]
    ):
        module.connection_map = conv_connection_map(module, input_.shape[1:], output.shape[1:])
    # Mean is across batches and timesteps
    module.layer_synops_per_timestep = (input_ * module.connection_map).mean(0).sum()

def linear_layer_synops_hook(module: nn.Linear, input_: List[torch.Tensor], output: torch.Tensor):
    input_ = _extract_single_input(input_)
    # Mean is across batches and timesteps
    module.layer_synops_per_timestep = input_.mean(0).sum() * module.out_features


@dataclass
class ModelSynopsHook:
    dt: Optional[float] = None

    def __call__(self, module: nn.Sequential, input_: Any, output: Any):
        scale_factors = []
        for module in module:
            if isinstance(module, nn.AvgPool2d):
                # Average pooling scales down the number of counted synops due to the averaging.
                # We need to correct for that by accumulating the scaling factors and multiplying
                # them to the counted Synops in the next conv or linear layer
                if module.kernel_size != module.stride:
                    warnings.warn(
                        f"In order for the Synops counter to work accurately the pooling "
                        f"layers kernel size should match their strides. At the moment at layer {name}, "
                        f"the kernel_size = {module.kernel_size}, the stride = {module.stride}."
                    )
                ks = module.kernel_size
                scaling = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
                scale_factors.append(scaling)
            if hasattr(module, "weight"):
                if hasattr(module, "layer_synops_per_timestep"):
                    # Multiply all scale factors (or use 1 if empty)
                    scaling = reduce(lambda x, y: x*y, scale_factors, 1)
                    module.synops_per_timestep = module.layer_synops_per_timestep * scaling
                    if self.dt is not None:
                        module.synops_per_second = module.synops_per_timestep / self.dt

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