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
        ...

    @classmethod
    @abstractmethod
    def monitor_layers(cls, config, layers: List[int]):
        ...

    @classmethod
    def get_valid_mapping(cls, model: "DynapcnnCompatibleNetwork") -> List[int]:
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
        is_valid, message = cls.get_samna_module().validate_configuration(config)
        if not is_valid:
            print(message)
        return is_valid

    @classmethod
    @abstractmethod
    def get_output_buffer(cls):
        ...