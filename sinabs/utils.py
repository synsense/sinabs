import torch
import torch.nn as nn
import numpy as np
from typing import List


def get_activations(torchanalog_model, tsrData, name_list=None):
    """
    Return torch analog model activations for the specified layers
    """
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
) -> [np.ndarray]:
    """
    Returns the activity of neurons in each layer of the network

    :param model: Model for which the activations are to be read out
    :param inp: Input to the model
    :param bRate: If true returns the rate, else returns spike count
    :param name_list: list of all layers whose activations need to be compared
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
    """
    Rescale the weights of the network, such that the activity of each specified layer is normalized.

    The method implemented here roughly follows the paper:
    `Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification` by Rueckauer et al.
    https://www.frontiersin.org/article/10.3389/fnins.2017.00682

    Args:
         ann(nn.Module): Torch module
         sample_data (nn.Tensor): Input data to normalize the network with
         output_layers (List[str]): List of layers to verify activity of normalization. Typically this is a relu layer
         param_layers (List[str]): List of layers whose parameters preceed `output_layers`
         percentile (float): A number between 0 and 100 to determine activity to be normalized by.
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
