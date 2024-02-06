from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from sinabs.layers import SqueezeMixin, StatefulLayer


def _extract_single_input(input_data: List[Any]) -> Any:
    """Extract single element of a list.

    Parameters:
        input_data: List that should have only one element

    Returns:
        The only element from the list

    Raises:
        ValueError if input_data does not have exactly
        one element.
    """
    if len(input_data) != 1:
        raise ValueError("Multiple inputs not supported for `input_diff_hook`")
    return input_data[0]


def conv_connection_map(
    layer: nn.Conv2d,
    input_shape: torch.Size,
    output_shape: torch.Size,
    device: Union[None, torch.device, str] = None,
) -> torch.Tensor:
    """Generate connectivity map for a convolutional layer The map indicates for each element in
    the layer input to how many postsynaptic neurons it connects (i.e. the fanout)

    Parameters:
        layer: Convolutional layer for which connectivity map is to be
               generated
        input_shape: Shape of the input data (N, C, Y, X)
        output_shape: Shape of layer output given `input_shape`
        device: Device on which the connectivity map should reside.
                Should be the same as that of the input to `layer`.
                If None, will select device of the weight of `layer`.

    Returns:
        torch.Tensor: Connectivity map indicating the fanout for each
                      element in the input
    """
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
    connection_map = deconvolve(output_ones, output_size=(1, *input_shape)).detach()
    connection_map.requires_grad = False
    if device is None:
        # If device is not specified, map to weight device
        connection_map = connection_map.to(layer.weight.device)
    else:
        connection_map = connection_map.to(torch.device(device))

    return connection_map


def get_hook_data_dict(module: nn.Module) -> Dict:
    """Convenience function to get `hook_data` attribute of a module if it has one and create it
    otherwise.

    Parameters:
        module: The module whose `hook_data` dict is to be fetched.
                If it does not have an attribute of that name, it
                will add an empty dict.
    Returns:
        The `hook_data` attribute of `module`. Should be a Dict.
    """
    if not hasattr(module, "hook_data"):
        module.hook_data = dict()
    return module.hook_data


def input_diff_hook(
    module: Union[nn.Conv2d, nn.Linear],
    input_: List[torch.Tensor],
    output: torch.Tensor,
):
    """Forwared hook to be registered with a Conv2d or Linear layer.

    Calculate the difference between the output if all weights were
    positive and the absolute of the actual output. Regularizing this
    value during training of an SNN can help reducing discrepancies
    between simulation and deployment on asynchronous processors.

    The hook should be registered with the layer using
    `torch.register_forward_hook`. It will be called automatically
    at each forward pass. Afterwards the data can be accessed with
    `module.hook_data['diff_output']`

    Parameters:
        module: Either a torch.nn.Conv2d or Linear layer
        input_: List of inputs to the layer. Should hold a single tensor.
        output: The layer's output.
    Effect:
        If `module` does not already have a `hook_data` attribute, it
        will be added and the difference value described above  will be
        stored under the key 'diff_output'. It is a tensor of the same
        shape as `output`.
    """
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
    """Forwared hook to be registered with a spiking sinabs layer.

    Calculate the mean firing rate per neuron per timestep.

    The hook should be registered with the layer using
    `torch.register_forward_hook`. It will be called automatically
    at each forward pass. Afterwards the data can be accessed with
    `module.hook_data['firing_rate']`

    Parameters:
        module: A spiking sinabs layer, such as `IAF` or `LIF`.
        input_: List of inputs to the layer. Ignored here.
        output: The layer's output.
    Effect:
        If `module` does not already have a `hook_data` attribute, it
        will be added and the mean firing rate will be stored under the
        key 'firing_rate'. It is a scalar value.
    """
    data = get_hook_data_dict(module)
    data["firing_rate"] = output.mean()


def firing_rate_per_neuron_hook(
    module: StatefulLayer, input_: Any, output: torch.Tensor
):
    """Forwared hook to be registered with a spiking sinabs layer.

    Calculate the mean firing rate per timestep for each neuron.

    The hook should be registered with the layer using
    `torch.register_forward_hook`. It will be called automatically
    at each forward pass. Afterwards the data can be accessed with
    `module.hook_data['firing_rate_per_neuron']`

    Parameters:
        module: A spiking sinabs layer, such as `IAF` or `LIF`.
        input_: List of inputs to the layer. Ignored here.
        output: The layer's output.
    Effect:
        If `module` does not already have a `hook_data` attribute, it
        will be added and the mean firing rate will be stored under the
        key 'firing_rate_per_neuron'. It is a tensor of the same
        shape as neurons of the spiking layer.
    """
    data = get_hook_data_dict(module)
    if isinstance(module, SqueezeMixin):
        # Common dimension for batch and time
        data["firing_rate_per_neuron"] = output.mean(0)
    else:
        # Output is of shape (N, T, ...)
        data["firing_rate_per_neuron"] = output.mean((0, 1))


def conv_layer_synops_hook(
    module: nn.Conv2d, input_: List[torch.Tensor], output: torch.Tensor
):
    """Forwared hook to be registered with a Conv2d layer.

    Calculate the mean synaptic operations per timestep.
    To be clear: Synaptic operations are summed over neurons, but
    averaged across batches / timesteps.
    Note that the hook assumes spike counts as inputs.
    Preceeding average pooling layers, which scale the data, might
    lead to false results and should be accounted for externally.

    The hook should be registered with the layer using
    `torch.register_forward_hook`. It will be called automatically
    at each forward pass. Afterwards the data can be accessed with
    `module.hook_data['layer_synops_per_timestep']`

    Parameters:
        module: A torch.nn.Conv2d layer
        input_: List of inputs to the layer. Must contain exactly one tensor
        output: The layer's output.
    Effect:
        If `module` does not already have a `hook_data` attribute, it
        will be added and the mean firing rate will be stored under the
        key 'layer_synops_per_timestep'. It is a scalar value.
        It will also store a connectivity map under the key 'connection_map',
        which holds the fanout for each input neuron.
    """
    data = get_hook_data_dict(module)
    input_ = _extract_single_input(input_)
    if (
        "connection_map" not in data
        # Ignore batch/time dimension when checking connectivity
        or data["connection_map"].shape[1:] != input_.shape[1:]
        or data["connection_map"].device != input_.device
    ):
        new_connection_map = conv_connection_map(
            module, input_.shape[1:], output.shape[1:], input_.device
        )
        data["connection_map"] = new_connection_map
    # Mean is across batches and timesteps
    data["layer_synops_per_timestep"] = (input_ * data["connection_map"]).mean(0).sum()


def linear_layer_synops_hook(
    module: nn.Linear, input_: List[torch.Tensor], output: torch.Tensor
):
    """Forwared hook to be registered with a Linear layer.

    Calculate the mean synaptic operations per timestep.
    To be clear: Synaptic operations are summed over neurons, but
    averaged across batches / timesteps.
    Note that the hook assumes spike counts as inputs.
    Preceeding average pooling layers, which scale the data, might
    lead to false results and should be accounted for externally.

    The hook should be registered with the layer using
    `torch.register_forward_hook`. It will be called automatically
    at each forward pass. Afterwards the data can be accessed with
    `module.hook_data['layer_synops_per_timestep']`

    Parameters:
        module: A torch.nn.Linear layer.
        input_: List of inputs to the layer. Must contain exactly one tensor
        output: The layer's output.
    Effect:
        If `module` does not already have a `hook_data` attribute, it
        will be added and the mean firing rate will be stored under the
        key 'layer_synops_per_timestep'.
    """
    data = get_hook_data_dict(module)
    input_ = _extract_single_input(input_)
    # Mean is across batches and timesteps
    synops = input_.mean(0).sum() * module.out_features
    data["layer_synops_per_timestep"] = synops


@dataclass
class ModelSynopsHook:
    """Forwared hook to be registered with a Sequential.

    Calculate the mean synaptic operations per timestep for the
    Conv2d and Linear layers inside the Sequential.
    Synaptic operations are summed over neurons, but averaged across
    batches / timesteps.
    Other than the layer-wise synops hook, this hook accounts for
    preceeding average pooling layers, which scale the data.

    To use this hook, the `conv_layer_synops_hook` and
    `linear_layer_synops_hook` need to be registered with the layers
    inside the Sequential first.  The hook should then be instantiated
    with or without a `dt` and registered with the Sequential using
    `torch.register_forward_hook`.
    Alternatively, refer to the function `register_synops_hooks` for
    a more convenient way of setting up the hooks.

    The hook will be called automatically at each forward pass. Afterwards
    the data can be accessed in several ways:

    - Each layer that has a synops hook registered, will have an entry
      'synops_per_timestep' in its `hook_data`. Other than the
      'layer_synops_per_timestep', this entry takes preceding average
      pooling layers into account.

    - The same values can be accessed through a dict inside the `hook_data`
      of the Sequential, under the key `synops_per_timestep`. The keys
      inside this dict correspond to the layer indices within the Sequential,
      e.g.: `sequential.hook_data['synops_per_timestep'][1]`

    - The `hook_data` of the sequential also contains a scalar entry
      'total_synops_per_timestep' which sums the synops over all layers.

    - If `dt` is not None, for each of the entries listed above, there
      will be a corresponding '(total_)synops_per_second' entry, indicating
      the synaptic operations per second, under the assumption that `dt`
      is the time step in seconds.

    Parameters:
      dt: If not None, should be a float that indicates the simulation
          time step in seconds. The synaptic operations will be also
          provided in terms of synops per second.
    """

    dt: Optional[float] = None

    def __call__(self, module: nn.Sequential, input_: Any, output: Any):
        """Forward call of the synops model hook. Should not be called manually but only by PyTorch
        during a forward pass.

        Parameters:
            module: A torch.nn.Sequential
            input_: List of inputs to the module.
            output: The module output.
        Effect:
            If `module` does not already have a `hook_data` attribute, it
            will be added and synaptic operations will be calculated and logged
            for all layers that have a layer-level synops hook registered.
        """
        module_data = get_hook_data_dict(module)
        module_data["total_synops_per_timestep"] = 0.0
        module_data["synops_per_timestep"] = dict()
        if self.dt is not None:
            module_data["total_synops_per_second"] = 0.0
            module_data["synops_per_second"] = dict()

        scale_factors = []
        for lyr_idx, lyr in enumerate(module):
            if isinstance(lyr, nn.AvgPool2d):
                # Average pooling scales down the number of counted synops due to the averaging.
                # We need to correct for that by accumulating the scaling factors and multiplying
                # them to the counted Synops in the next conv or linear layer
                if lyr.kernel_size != lyr.stride:
                    warnings.warn(
                        "In order for the Synops counter to work accurately the pooling "
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
                    layer_data = lyr.hook_data
                    # Multiply all scale factors (or use 1 if empty)
                    scaling = reduce(lambda x, y: x * y, scale_factors, 1)
                    synops = layer_data["layer_synops_per_timestep"] * scaling
                    layer_data["synops_per_timestep"] = synops
                    module_data["synops_per_timestep"][lyr_idx] = synops
                    module_data["total_synops_per_timestep"] += synops
                    if self.dt is not None:
                        synops_per_sec = layer_data["synops_per_timestep"] / self.dt
                        layer_data["synops_per_second"] = synops_per_sec
                        module_data["synops_per_second"][lyr_idx] = synops_per_sec
                        module_data["total_synops_per_second"] += synops_per_sec

                # For any module with weight: Reset `scale_factors` even if it doesn't count synops
                scale_factors = []


def register_synops_hooks(module: nn.Sequential, dt: Optional[float] = None):
    """Convenience function to register all the necessary hooks to collect synops statistics in a
    sequential model.

    This can be used instead of calling the torch function
    `register_forward_hook` on all layers.

    Parameters:
        module: Sequential model for which the hooks should be registered.
        dt: If not None, should be a float indicating the simulation
            time step in seconds. Will also calculate synaptic operations per second.
    """
    for lyr in module:
        if isinstance(lyr, nn.Conv2d):
            lyr.register_forward_hook(conv_layer_synops_hook)
        elif isinstance(lyr, nn.Linear):
            lyr.register_forward_hook(linear_layer_synops_hook)
    model_hook = ModelSynopsHook(dt)
    module.register_forward_hook(model_hook)
