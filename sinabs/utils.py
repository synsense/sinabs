from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

import sinabs
from .validate_memory_speck import ValidateMapping


def reset_states(model: nn.Module) -> None:
    """Helper function to recursively reset all states of spiking layers within the model.

    Parameters:
        model: The torch module
    """
    for layer in model.children():
        if len(list(layer.children())):
            reset_states(layer)
        elif isinstance(layer, sinabs.layers.StatefulLayer):
            layer.reset_states()


def zero_grad(model: nn.Module) -> None:
    """Helper function to recursively zero the gradients of all spiking layers within the model.

    Parameters:
        model: The torch module
    """
    for layer in model.children():
        if len(list(layer.children())):
            zero_grad(layer)
        elif isinstance(layer, sinabs.layers.StatefulLayer):
            layer.zero_grad()


def get_activations(torchanalog_model, tsrData, name_list=None):
    """Return torch analog model activations for the specified layers."""
    torch_modules = dict(torchanalog_model.named_modules())

    # Populate layer names
    if name_list is None:
        name_list = ["Input"] + list(torch_modules.keys())[1:]

    analog_activations = []

    for layer_name in name_list:
        if layer_name == "Input":
            # Bypass input layers
            arrOut = tsrData.detach().cpu().numpy()
            analog_activations.append(arrOut)

    # Define hook
    def hook(module, inp, output):
        arrOut = output.detach().cpu().numpy()
        analog_activations.append(arrOut)

    hooklist = []
    # Attach and register hook
    for layer_name in name_list:
        if layer_name == "Input":
            continue
        torch_layer = torch_modules[layer_name]
        hooklist.append(torch_layer.register_forward_hook(hook))

    # Do a forward pass
    with torch.no_grad():
        torchanalog_model.eval()
        torchanalog_model(tsrData)

    # Remove hooks
    for h in hooklist:
        h.remove()

    return analog_activations


def get_network_activations(
    model: nn.Module, inp, name_list: List = None, bRate: bool = False
) -> List[np.ndarray]:
    """Returns the activity of neurons in each layer of the network.

    Parameters:
        model: Model for which the activations are to be read out
        inp: Input to the model
        bRate: If true returns the rate, else returns spike count
        name_list: list of all layers whose activations need to be compared
    """
    spike_counts = []
    tSim = len(inp)

    # Define hook
    def hook(module, inp, output):
        arrOut = output.float().sum(0).cpu().numpy()
        spike_counts.append(arrOut)

    # Generate default list of layers
    if name_list is None:
        name_list = ["Input"] + [lyr.layer_name for lyr in model.layers]

    # Extract activity for each layer of interest
    for layer_name in name_list:
        # Append input activity
        if layer_name == "Input":
            spike_counts.append(inp.float().sum(0).cpu().numpy() * 1000)
        else:
            # Activity of other layers
            lyr = dict(model.named_modules())[layer_name]
            lyr.register_forward_hook(hook)

    with torch.no_grad():
        model(inp)

    if bRate:
        spike_counts = [(counts / tSim * 1000) for counts in spike_counts]
    return spike_counts


def normalize_weights(
    ann: nn.Module,
    sample_data: torch.Tensor,
    output_layers: List[str],
    param_layers: List[str],
    percentile: float = 99,
):
    """Rescale the weights of the network, such that the activity of each specified layer is
    normalized.

    The method implemented here roughly follows the paper:
    `Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification` by Rueckauer et al.
    https://www.frontiersin.org/article/10.3389/fnins.2017.00682

    Parameters:
         ann: Torch module
         sample_data: Input data to normalize the network with
         output_layers: List of layers to verify activity of normalization. Typically this is a relu layer
         param_layers: List of layers whose parameters preceed `output_layers`
         percentile: A number between 0 and 100 to determine activity to be normalized by
                     where a 100 corresponds to the max activity of the network. Defaults to 99.
    """
    # Network activity storage
    output_data = []

    # Hook to save data
    def save_data(lyr, input, output):
        output_data.append(output.clone())

    # All the named layers of the module
    named_layers = dict(ann.named_children())

    for i in range(len(output_layers)):
        param_layer = named_layers[param_layers[i]]
        output_layer = named_layers[output_layers[i]]

        handle = output_layer.register_forward_hook(save_data)

        with torch.no_grad():
            _ = ann(sample_data)

            # Get max output
            max_lyr_out = np.percentile(output_data[-1].cpu().numpy(), percentile)

            # Rescale weights to normalize max output
            for p in param_layer.parameters():
                p.data *= 1 / max_lyr_out

        output_data.clear()
        # Deregister hook
        handle.remove()


def set_batch_size(model: nn.Module, batch_size: int):
    """Update any model with sinabs squeeze layers to a given batch size.

    Args:
        model (nn.Module): pytorch model with sinabs Squeeze layers
        batch_size (int): The new batch size
    """
    for mod in model.modules():
        if isinstance(mod, sinabs.layers.SqueezeMixin):
            mod.batch_size = batch_size
            # reset_states(mod)


def validate_memory_mapping_speck(
    input_feature_size: int,
    output_feature_size: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    input_dimension: Tuple[int, int] = [64, 64],
    conv_2d: bool = True,
):
    """Helper function to verify if it is possible to map a specific layer on to speck.
        This function validates kernel and neuron memories. It doesnt check for all the network layers together.
        It considers the mapping of a Conv2D layer only.

    Args:
        input_feature_size (int): number of input channels
        output_feature_size (int): number of output channels
        kernel_size (Tuple[int, int]): 2D kernel size
        stride (Tuple[int, int]): 2D stride size
        padding (Tuple[int, int]): 2D padding size
        input_dimension (Tuple[int, int]): 2D input dimension size. Defaults to [64,64]
        conv_2d (bool): if it is mapping a Conv2D layers. Defaults to True. It won't validate other types of network.

    Return:
        msg (string): Message indicating layer can be mapped with total size of kernel and neuron memories.

    Raises:
        Exception: if neuron or kernel memories are higher than available on chip.
    """

    if not conv_2d:
        raise ValueError("This function only validates Conv2D layers.")

    validate = ValidateMapping(
        input_feature_size,
        output_feature_size,
        kernel_size,
        stride,
        padding,
        [input_dimension[0], input_dimension[1]],
        conv_2d,
    )
    (
        kernel,
        neuron,
        kernel_error_msg,
        neuron_error_msg,
    ) = validate.calculate_total_memory()

    if kernel_error_msg != "" or neuron_error_msg != "":
        raise Exception(kernel_error_msg + neuron_error_msg)
    else:
        msg = (
            "Layer can be mapped successfully. "
            f"Kernel memory is {kernel:g}Ki and neuron memory is {neuron:g}Ki."
        )

    return msg
