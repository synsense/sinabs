import samna
from typing import List
from samna.speck2f.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
from sinabs.backend.dynapcnn.utils import extend_readout_layer

from .dynapcnn import DynapcnnConfigBuilder

# Since most of the configuration is identical to DYNAP-CNN, we can simply inherit this class


class Speck2FConfigBuilder(DynapcnnConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.speck2f

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2f_event_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2f_event_output_event()

    @classmethod
    def get_dynapcnn_layer_config_dict(cls, layer: DynapcnnLayer):
        config_dict = super().get_dynapcnn_layer_config_dict(layer=layer)
        config_dict.pop("weights_kill_bit")
        config_dict.pop("biases_kill_bit")
        config_dict.pop("neurons_value_kill_bit")
        return config_dict

    @classmethod
    def build_config(cls, model: "DynapcnnNetwork", chip_layers: List[int]):
        if model._is_chip_layers_ordering_auto:
            print(f"Extend readout layer for Speck2E")
            model = extend_readout_layer(
                model
            )  # extend only if chip layers ordering is determined automatically
            chip_layers = cls.get_valid_mapping(model=model)
        return super().build_config(model, chip_layers)
