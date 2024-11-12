from abc import ABC, abstractmethod
from typing import Dict, List

import samna
from samna.dynapcnn.configuration import DynapcnnConfiguration

import sinabs
import sinabs.backend
import sinabs.backend.dynapcnn

from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .exceptions import InvalidModel
from .mapping import LayerConstraints, get_valid_mapping


class ConfigBuilder(ABC):
    @classmethod
    @abstractmethod
    def get_samna_module(self):
        """Get the saman parent module that hosts all the appropriate sub-modules and classes.

        Returns
        -------
        samna module
        """

    @classmethod
    @abstractmethod
    def get_default_config(cls):
        """
        Returns
        -------
        Returns the default configuration for the device type
        """

    @classmethod
    @abstractmethod
    def build_config(
        cls,
        layers: Dict[int, DynapcnnLayer],
        destination_map: Dict[int, List[int]],
        layer2core_map: Dict[int, int],
    ) -> DynapcnnConfiguration:
        """Build the configuration given a model.

        Parameters
        ----------
        model:
            The target model
        chip_layers:
            Chip layers where the given model layers are to be mapped.

        Returns
        -------
            Samna Configuration object
        """

    @classmethod
    @abstractmethod
    def get_constraints(cls) -> List[LayerConstraints]:
        """Returns the layer constraints of a the given device.

        Returns
        -------
            List[LayerConstraints]
        """

    @classmethod
    @abstractmethod
    def monitor_layers(cls, config, layers: List[int]):
        """Enable the monitor for a given set of layers in the config object."""

    @classmethod
    def map_layers_to_cores(cls, layers: Dict[int, DynapcnnLayer]) -> Dict[int, int]:
        """Find a mapping from DynapcnnLayers onto on-chip cores

        Parameters
        ----------
        - layers: Dict with layer indices as keys and DynapcnnLayer instances as values

        Returns
        -------
        - Dict mapping layer indices (keys) to assigned core IDs (values).
        """

        return get_valid_mapping(layers, cls.get_constraints())

    @classmethod
    def validate_configuration(cls, config) -> bool:
        """Check if a given configuration is valid.

        Parameters
        ----------
        config:
            Configuration object

        Returns
        -------
        True if the configuration is valid, else false
        """
        is_valid, message = cls.get_samna_module().validate_configuration(config)
        if not is_valid:
            print(message)
        return is_valid

    @classmethod
    @abstractmethod
    def get_input_buffer(cls):
        """Initialize and return the appropriate output buffer object Note that this just the
        buffer object.

        This does not actually connect the buffer object to the graph. (It is needed as of samna
        0.21.0)
        """

    @classmethod
    @abstractmethod
    def get_output_buffer(cls):
        """Initialize and return the appropriate output buffer object Note that this just the
        buffer object.

        This does not actually connect the buffer object to the graph.
        """

    @classmethod
    @abstractmethod
    def reset_states(cls, config, randomize=False):
        """Randomize or reset the neuron states.

        Parameters
        ----------
            randomize (bool):
                If true, the states will be set to random initial values. Else they will be set to zero
        """

    @classmethod
    def set_all_v_mem_to_zeros(cls, samna_device, layer_id: int) -> None:
        """Reset all memory states to zeros.

        Parameters
        ----------
        samna_device:
            samna device object to erase vmem memory.
        layer_id:
            layer index
        """
        mod = cls.get_samna_module()
        layer_constraint: LayerConstraints = cls.get_constraints()[layer_id]
        events = []
        for i in range(layer_constraint.neuron_memory):
            event = mod.event.WriteNeuronValue()
            event.address = i
            event.layer = layer_id
            event.neuron_state = 0
            events.append(event)

        temporary_source_node = cls.get_input_buffer()
        temporary_graph = samna.graph.sequential(
            [temporary_source_node, samna_device.get_model().get_sink_node()]
        )
        temporary_graph.start()
        temporary_source_node.write(events)
        temporary_graph.stop()
        return
