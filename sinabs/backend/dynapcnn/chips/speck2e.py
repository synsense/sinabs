from typing import Dict

import samna
from samna.speck2e.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer

from .dynapcnn import DynapcnnConfigBuilder


# Inherit DynapCNNConfigBuilder to share implementation with other DynapCNN/Speck devices
class Speck2EConfigBuilder(DynapcnnConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.speck2e

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_dvs_layer_config(cls):
        return SpeckConfiguration().DVSLayerConfig

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2e_event_speck2e_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2e_event_output_event()
