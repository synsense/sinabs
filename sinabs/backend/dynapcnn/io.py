import torch
import time
import samna
from samna.dynapcnn.event import RouterEvent, Spike
from typing import List


samna_node = None


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


def get_all_samna_devices():
    """
    Returns a list of all samna devices

    Returns
    -------
    devices:
        A list of samna devices currently connected
    """
    global samna_node

    if samna_node is None:
        # initialize the main SamnaNode
        receiver_endpoint = "tcp://0.0.0.0:33335"
        sender_endpoint = "tcp://0.0.0.0:33336"
        node_id = 1
        interpreter_id = 2
        samna_node = samna.SamnaNode(sender_endpoint, receiver_endpoint, node_id)

        # setup the python interpreter node
        samna.setup_local_node(receiver_endpoint, sender_endpoint, interpreter_id)

        # open a connection to device_node
        samna.open_remote_node(node_id, "device_node")

    # Find device
    return samna.device_node.DeviceController.get_unopened_devices()


def is_device_type(dev_info: samna.device.DeviceInfo, dev_type: str) -> bool:
    """
    Check if a DeviceInfo object is of a given device type 'dev_type'
    Parameters
    ----------
    dev_info: samna.device.DeviceInfo
        Device info object
    dev_type: str
        Device type as a string

    Returns
    -------
    boolean

    """
    return dev_info.device_type_name == dev_type


def discover_device(device_id: str):
    """
    Discover a samna device

    Parameters
    ----------
    device_id: str
        Device name/identifier (DynapcnnDevKit:0 or speck:0 or DVXplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    Returns
    -------
    device_info: samna.device.DeviceInfo

    See Also
    --------
    get_device

    """

    synsense_devs = get_all_samna_devices()

    device_name, device_num = device_id.split(":")
    device_num = int(device_num)

    device_info = [x for x in synsense_devs if is_device_type(x, device_name)][device_num]
    return device_info


def open_device(device_id: str):
    """

    Parameters
    ----------
    device_id

    Returns
    -------

    """
    device_name, device_num = device_id.split(":")
    device_num = int(device_num)

    dev_info = discover_device(device_id)
    # Open Devkit
    samna.device_node.DeviceController.open_device(dev_info, f"{device_name}_{device_num}")

    # get the handle of our dev-kit
    device = samna.device_node.__dict__[f"{device_name}_{device_num}"]
    return device


def close_device(device_id: str):
    """
    Convenience method to close a device
    Parameters
    ----------
    device_id: str
        Device name/identifier (dynapcnn:0 or speck:0 or dynapcnn:1 ... )

    """
    device_name, device_num = device_id.split(":")
    samna.device_node.DeviceController.close_device(f"{device_name}_{device_num}")