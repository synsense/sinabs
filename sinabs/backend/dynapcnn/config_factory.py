from typing import List
from .io import _parse_device_string
from .config_builder import ConfigBuilder
from .chips import *
import samna


class ChipFactory:
    device_name: str
    device_id: int

    def __init__(self, device_str: str):
        """
        Factory class to access config builder and other device specific methods

        Parameters
        ----------
        device_str
        """
        self.device_name, self.device_id = _parse_device_string(device_str)

    def get_config_builder(self) -> ConfigBuilder:
        if self.device_name == "dynapcnndevkit":
            return DynapcnnConfigBuilder()
        elif self.device_name == "speck2":
            return Speck2ConfigBuilder()
        elif self.device_name == "speck2b":
            return Speck2BConfigBuilder()
        else:
            raise Exception(f"Builder not found for device type: {self.device_name}")

    def xytp_to_events(self) -> List["Spike"]:
        raise NotImplementedError
