import torch
import time
import samna
from samna.dynapcnn.event import RouterEvent, Spike
from typing import List


def raster_to_events(raster: torch.Tensor, layer: int = 0) -> List[RouterEvent]:
    """
    Convert spike raster to events for dynapcnn

    Parameters:
    -----------

    raster: torch.Tensor
        A 4 dimensional tensor of spike events with the dimensions [Time, Channel, Height, Width]

    layer: int
        The index of the layer to route the events to

    Returns:
    --------

    events: List[RouterEvent]
        A list of events that will be streamed to the device
    """
    t, ch, y, x = torch.where(raster)
    evData = torch.stack((t, ch, y, x), dim=0).T
    events = []
    for row in evData:
        ev = RouterEvent()
        ev.layer = layer
        ev.x = row[3]
        ev.y = row[2]
        ev.feature = row[1]
        # ev.timestamp = row[0]
        events.append(ev)
    return events


def events_to_raster(eventList: List, layer: int = 0) -> torch.Tensor:
    """
    Convert an eventList read from `samna` to a tensor `raster` by filtering only the events specified by `layer`

    Parameters:
    -----------

    eventList: List
        A list comprising of events from samna API

    layer: int
        The index of layer for which the data needs to be converted

    Returns:
    --------

    raster: torch.Tensor
    """
    evsFiltered = []
    for ev in eventList:
        if isinstance(ev, Spike):
            if ev.layer == layer:
                evsFiltered.append((ev.timestamp, ev.feature, ev.x, ev.y))

    return evsFiltered


# def connect():
#     """
#     Connect to device through samna
#
#     Returns:
#     --------
#     dev_kit:
#         Handler object to the development kit
#     """
#     samna_node = samna.SamnaNode()
#
#     # Launch  dev_kit
#     samna_node.open_dev_kit()
#     time.sleep(1)
#     r = samna.connect()
#
#     time.sleep(1)
#     dk = r.SpeckDevKit
#     dk.start_reader_writer()
#     io = dk.get_io_module()
#     io.write_config(0x2, 0xF)
#     time.sleep(1)
#     io.write_config(0x0009, 0x001F)
#     time.sleep(1)
#     io.write_config(0x0009, 0x0000)
#     io.deassert_reset()
#
#     dk = r.SpeckDevKit
#
#     return dk, samna_node
