import torch.nn as nn
from sinabs import Network
from sinabs.layers import TorchLayer
from typing import Dict, Union
import samna


def to_speck_config(network: Union[nn.Module, TorchLayer]) -> Dict:
    """
    Build a configuration object of a given module

    :param network: sinabs.Network or sinabs.layers.TorchLayer instance
    """
    config = {}
    if isinstance(network, Network):
        for layer in network.spiking_model.children():
            config[layer.layer_name] = to_speck_config(layer)
            # Populate source/destination layers here
            # Consolidate pooling and conv layers here
    elif isinstance(network, TorchLayer):
        #TODO: Do your thing for config
        return config


def write_to_device(config: Dict, device: samna.SpeckModel, weights=None):
    """
    Write your model configuration to dict

    :param config:
    :param device:
    :return:
    """
    device.set_config(to_speck_config(config))
    if weights:
        device.set_weights(weights)
    device.apply()


def to_speck_config(config: Dict) -> samna.SpeckConfig:
    speck_config = samna.SpeckConfig()
    # TODO

    # Populate the config
    return speck_config

