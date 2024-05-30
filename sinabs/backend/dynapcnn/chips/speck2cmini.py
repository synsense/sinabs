from typing import List, Dict

import samna
from samna.speck2cMini.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
from sinabs.backend.dynapcnn.mapping import LayerConstraints

from .dynapcnn import DynapcnnConfigBuilder

# Since most of the configuration is identical to DYNAP-CNN, we can simply inherit this class


class Speck2CMiniConfigBuilder(DynapcnnConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.speck2cMini

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2c_mini_event_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2c_mini_event_output_event()

    @classmethod
    def get_dynapcnn_layer_config_dict(cls, layer: DynapcnnLayer, layers_mapper: Dict[int, DynapcnnLayer]) -> dict:
        config_dict = super().get_dynapcnn_layer_config_dict(layer=layer, layers_mapper=layers_mapper)
        config_dict.pop("weights_kill_bit")
        config_dict.pop("biases_kill_bit")
        config_dict.pop("neurons_value_kill_bit")
        return config_dict

    @classmethod
    def get_constraints(cls) -> List[LayerConstraints]:
        weights_memory_size = [
            16 * 1024,  # 0
            32 * 1024,  # 1
            32 * 1024,  # 2
            64 * 1024,  # 3
            16 * 1024,  # 4
        ]

        neurons_memory_size = [
            64 * 1024,  # 0
            32 * 1024,  # 1
            32 * 1024,  # 2
            16 * 1024,  # 3
            16 * 1024,  # 4
        ]

        bias_memory_size = [0, 0, 0, 1024, 1024]  # 0  # 1  # 2  # 3  # 4

        constraints = [
            LayerConstraints(km, nm, bm)
            for (km, nm, bm) in zip(
                weights_memory_size, neurons_memory_size, bias_memory_size
            )
        ]
        return constraints
