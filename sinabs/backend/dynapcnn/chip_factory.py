import torch
import numpy as np
from typing import List, Tuple, Optional
from .utils import _parse_device_string
from .config_builder import ConfigBuilder
from .chips import *


class ChipFactory:

    supported_devices = {
        "dynapcnndevkit": DynapcnnConfigBuilder,
        "speck2b": Speck2BConfigBuilder,
        "speck2btiny": Speck2BConfigBuilder, # It is the same chip, so doesn't require a separate builder
        "speck2cmini": Speck2CMiniConfigBuilder,
        "speck2dmini": Speck2DMiniConfigBuilder,
        "speck2e": Speck2EConfigBuilder,
        "speck2edevkit": Speck2EConfigBuilder,
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

    def raster_to_events(self, raster: torch.Tensor, layer, dt=1e-3, truncate: bool = False, delay_factor: float = 0) -> List:
        """
        Convert spike raster to events for DynapcnnNetworks

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

        delay_factor: float
            (default = 0) Start simulation from this time. (in seconds)


        Returns
        -------

        events: List[Spike]
            A list of events that will be streamed to the device
        """
        assert delay_factor >= 0.0, print("Delay factor cannot be a negative value!")
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
            ev.timestamp = int( ( row[0].item() * 1e6 * dt ) + ( delay_factor * 1e6 ) )  # Time in uS
            events.append(ev)
        return events

    def xytp_to_events(self, xytp: np.ndarray, layer, reset_timestamps, delay_factor: float = 0) -> List:
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

        delay_factor: float
            (default = 0) Start simulation from this time. (in seconds)

        Returns
        -------

        events: List[Spike]
            A list of events that will be streamed to the device
        """

        # Check delay factor as it being negative will crash the method.
        assert delay_factor >= 0, print("Delay factor cannot be a negative value!")

        # Check the smallest timestamp is larger or equal to zero to prevent overflows.
        tstart = xytp["t"].min()
        assert tstart >= 0, print("Timestamps cannot be negative values!")
        samna_module = self.get_config_builder().get_samna_module()
        # Get the appropriate Spike class
        Spike = samna_module.event.Spike

        events = []
        for row in xytp:
            ev = Spike()
            ev.layer = layer
            ev.x = row["x"]
            ev.y = row["y"]
            ev.feature = row["p"]
            if reset_timestamps:
                ev.timestamp = int(row["t"] - tstart + ( delay_factor * 1e6 ) )# Time in uS
            else:
                ev.timestamp = int(row["t"] + ( delay_factor * 1e6 ) )
            events.append(ev)
        return events

    def events_to_raster(self, events: List, dt: float =1e-3, shape: Optional[Tuple]=None) -> torch.Tensor:
        """
        Convert events from DynapcnnNetworks to spike raster

        Parameters
        ----------

        events: List[Spike]
            A list of events that will be streamed to the device
        dt: float
            Length of each time step for rasterization
        shape: Optional[Tuple]
            Shape of the raster to be produced, excluding the time dimension. (Channel, Height, Width)
            If this is not specified, the shape is inferred based on the max values found in the events.

        Returns
        -------
        raster: torch.Tensor
            A 4 dimensional tensor of spike events with the dimensions [Time, Channel, Height, Width]
        """
        timestamps = [event.timestamp for event in events]
        start_timestamp = min(timestamps)
        timestamps = [ts - start_timestamp for ts in timestamps]
        xs = [event.x for event in events]
        ys = [event.y for event in events]
        features = [event.feature for event in events]

        # Initialize an empty raster
        if shape:
            shape = (int(max(timestamps)*dt)+1, *shape)
            raster = torch.zeros(shape)
        else:
            raster = torch.zeros(int(max(timestamps)*dt)+1, max(features)+1, max(xs)+1, max(ys)+1)

        for event in events:
            raster[int((event.timestamp - start_timestamp)*dt), event.feature, event.x, event.y] += 1
        return raster

