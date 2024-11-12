from typing import Dict, List

import samna
from samna.speck2f.configuration import SpeckConfiguration

from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer

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
    def get_dynapcnn_layer_config_dict(
        cls,
        layer: DynapcnnLayer,
        layer2core_map: Dict[int, int],
        destination_indices: List[int],
    ) -> dict:
        """Generate config dict from DynapcnnLayer instance

        Parameters
        ----------
        - layer (DynapcnnLayer): Layer instance from which to generate the config
        - layer2core_map (Dict): Keys are layer indices, values are corresponding
            cores on hardware. Needed to map the destinations.]
        - destination_indices (List): Indices of destination layers for `layer`

        Returns
        -------
        - Dict that holds the information to configure the on-chip core
        """
        config_dict = super().get_dynapcnn_layer_config_dict(
            layer=layer,
            layer2core_map=layer2core_map,
            destination_indices=destination_indices,
        )
        config_dict.pop("weights_kill_bit")
        config_dict.pop("biases_kill_bit")
        config_dict.pop("neurons_value_kill_bit")
        return config_dict
