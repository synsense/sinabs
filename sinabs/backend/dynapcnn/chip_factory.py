import torch
import numpy as np
from typing import List
from .io import _parse_device_string
from .config_builder import ConfigBuilder
from .chips import *


class ChipFactory:

    supported_devices = {
        "dynapcnndevkit": DynapcnnConfigBuilder,
        "speck2b": Speck2BConfigBuilder,
    }

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
        if self.device_name not in self.supported_devices:
            raise Exception(f"Builder not found for device type: {self.device_name}")

    def get_config_builder(self) -> ConfigBuilder:
        return self.supported_devices[self.device_name]()

    def raster_to_events(self, raster: torch.Tensor, layer, dt=1e-3, truncate: bool = False) -> List:
        """
        Convert spike raster to events for DynaapcnnDevKit

        Parameters
        ----------

        raster: torch.Tensor
            A 4 dimensional tensor of spike events with the dimensions [Time, Channel, Height, Width]

        layer: int
            The index of the layer to route the events to

        dt: float
            Length of time step of the raster in seconds

        truncate: bool
            (default = False) Limit time-bins with more than one spikes to one spike.


        Returns
        -------

        events: List[Spike]
            A list of events that will be streamed to the device
        """
        samna_module = self.get_config_builder().get_samna_module()
        # Get the appropriate Spike class
        Spike = samna_module.event.Spike
        if truncate:
            t, ch, y, x = torch.where(raster)
            evData = torch.stack((t, ch, y, x), dim=0).T
        else:
            event_list = []
            max_raster = raster.max()
            for val in range(int(max_raster), 0, -1):
                t, ch, y, x = torch.where(raster == val)
                evData = torch.stack((t, ch, y, x), dim=0).T
                evData = evData.repeat(val, 1)
                event_list.extend(evData)
            evData = torch.stack(sorted(event_list, key=lambda event: event[0]), dim=0)

        events = []
        for row in evData:
            ev = Spike()
            ev.layer = layer
            ev.x = row[3]
            ev.y = row[2]
            ev.feature = row[1]
            ev.timestamp = int(row[0].item() * 1e6 * dt)  # Time in uS
            events.append(ev)
        return events

    def xytp_to_events(self, xytp: np.ndarray, layer, reset_timestamps) -> List:
        """
        Convert series of spikes in a structured array (eg. from aermanager) to events for DynaapcnnDevKit

        Parameters
        ----------

        xytp: torch.Tensor
            A numpy structured array with columns x, y, timestamp, polarity

        layer: int
            The index of the layer to route the events to

        reset_timestamps: Boolean

            If set to True, timestamps will be aligned to start from 0

        Returns
        -------

        events: List[Spike]
            A list of events that will be streamed to the device
        """
        samna_module = self.get_config_builder().get_samna_module()
        # Get the appropriate Spike class
        Spike = samna_module.event.Spike

        events = []
        tstart = xytp["t"].min()
        for row in xytp:
            ev = Spike()
            ev.layer = layer
            ev.x = row["x"]
            ev.y = row["y"]
            ev.feature = row["p"]
            if reset_timestamps:
                ev.timestamp = row["t"] - tstart# Time in uS
            else:
                ev.timestamp = row["t"]
            events.append(ev)
        return events
