import samna
from samna.speck2e.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer

from .dynapcnn import DynapcnnConfigBuilder

# Since most of the configuration is identical to DYNAP-CNN, we can simply inherit this class


class Speck2EConfigBuilder(DynapcnnConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.speck2e

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2e_event_speck2e_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2e_event_output_event()

    @classmethod
    def set_kill_bits(cls, layer: DynapcnnLayer, config_dict: dict) -> dict:
        return config_dict

    @classmethod
    def get_dynapcnn_layer_config_dict(cls, layer: DynapcnnLayer):
        config_dict = super().get_dynapcnn_layer_config_dict(layer=layer)
        return config_dict
