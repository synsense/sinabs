from abc import ABC, abstractmethod
from typing import List
from .mapping import LayerConstraints, get_valid_mapping


class ConfigBuilder(ABC):

    @classmethod
    @abstractmethod
    def get_samna_module(self):
        """
        Get the saman parent module that hosts all the appropriate sub-modules and classes

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
    def build_config(cls, model: "DynapcnnCompatibleNetwork", chip_layers: List[int]):
        """
        Build the configuration given a model

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
        """
        Returns the layer constraints of a the given device.

        Returns
        -------
            List[LayerConstraints]
        """

    @classmethod
    @abstractmethod
    def monitor_layers(cls, config, layers: List[int]):
        """
        Enable the monitor for a given set of layers in the config object
        """

    @classmethod
    def get_valid_mapping(cls, model: "DynapcnnCompatibleNetwork") -> List[int]:
        """
        Find a valid set of layers for a given model

        Parameters
        ----------
        model (DynapcnnCompatibleNetwork):
            A model

        Returns
        -------
        List of layer indices corresponding to each layer of the model
        """
        mapping = get_valid_mapping(model, cls.get_constraints())
        # turn the mapping into a dict
        mapping = {m[0]: m[1] for m in mapping}
        # apply the mapping
        chip_layers_ordering = [
            mapping[i] for i in range(len(model.compatible_layers))
        ]
        return chip_layers_ordering

    @classmethod
    def validate_configuration(cls, config) -> bool:
        """
        Check if a given configuration is valid

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
    def get_output_buffer(cls):
        """
        Initialize and return the appropriate output buffer object
        Note that this just the buffer object. This does not actually connect the buffer object to the graph.
        """