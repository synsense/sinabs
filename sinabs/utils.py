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
