from abc import ABC, abstractmethod
from typing import List
from .mapping import LayerConstraints


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
        ...

    @classmethod
    @abstractmethod
    def get_default_config(cls):
        ...

    @classmethod
    @abstractmethod
    def build_config(cls, model: "DynapcnnCompatibleNetwork", chip_layers: List[int]):
        ...

    @classmethod
    @abstractmethod
    def get_constriants(cls) -> List[LayerConstraints]:
        ...

    @classmethod
    @abstractmethod
    def monitor_layers(cls, config, layers: List[int]):
        ...

    @classmethod
    @abstractmethod
    def get_valid_mapping(cls, model: "DynapcnnCompatibleNetwork") -> List[int]:
        ...

    @classmethod
    @abstractmethod
    def validate_configuration(cls, config) -> bool:
        ...
