import warnings

import samna
import torch
from itertools import groupby
from typing import List, Dict
import numpy as np
from .utils import _parse_device_string

# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck": "speck",
    "speck2b": "Speck2bTestboard",
    "speck2devkit": "Speck2DevKit",
    "speck2btiny": "Speck2bDevKitTiny",
    "speck2e": "Speck2eTestBoard", # with a capital B for board
    "speck2edevkit": "Speck2eDevKit",
    "dynapse1devkit": "Dynapse1DevKit",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
    "pollendevkit": "PollenDevKit",
    "dynapcnndevkit": "DynapcnnDevKit",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
}

device_type_map = {v: k for (k, v) in device_types.items()}
device_map = {}

def enable_timestamps(
        device_id: str,
) -> None:
    """
    Disable timestamps of the samna node

    Args
    ----
    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_name, device_idx = _parse_device_string(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 1)
    else:
        device_handle.get_stop_watch().set_enable_value(True)


def disable_timestamps(
        device_id: str,
) -> None:
    """
    Disable timestamps of the samna node

    Args
    ----

    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_name, device_idx = _parse_device_string(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 0)
    else:
        device_handle.get_stop_watch().set_enable_value(False)


def reset_timestamps(
        device_id: str,
) -> None:
    """
    Disable timestamps of the samna node

    Args
    ----
    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_name, device_idx = _parse_device_string(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 1)
    else:
        device_handle.get_stop_watch().reset()


def events_to_raster(event_list: List, layer: int) -> torch.Tensor:
    """
    Convert an eventList read from `samna` to a tensor `raster` by filtering only the events specified by `layer`.

    Parameters
    ----------

    event_list: List
        A list comprising of events from samna API

    layer: int
        The index of layer for which the data needs to be converted

    Returns
    -------

    raster: torch.Tensor
    """
    evs_filtered = filter(lambda x: isinstance(x, samna.dynapcnn.event.Spike), event_list)
    evs_filtered = filter(lambda x: x.layer == layer, evs_filtered)
    raise NotImplementedError
    raster = map(rasterize, evs_filtered)
    return raster


def events_to_xytp(event_list: List, layer: int) -> np.array:
    """
    Convert an eventList read from `samna` to a numpy structured array of `x`, `y`, `t`, `channel`.

    Parameters
    ----------

    event_list: List
        A list comprising of events from samna API

    layer: int
        The index of layer for which the data needs to be converted

    Returns
    -------

    xytc: np.array
        A numpy structured array with columns `x`, `y`, `t`, `channel`.
    """
    evs_filtered = list(filter(lambda x: isinstance(x, samna.dynapcnn.event.Spike) and x.layer == layer, event_list))
    xytc = np.empty(
        len(evs_filtered),
        dtype=[("x", np.uint16), ("y", np.uint16), ("t", np.uint64), ("channel", np.uint16)]
    )

    for i, event in enumerate(evs_filtered):
        xytc[i]["x"] = event.x
        xytc[i]["y"] = event.y
        xytc[i]["t"] = event.timestamp
        xytc[i]["channel"] = event.feature
    return xytc


def get_device_map() -> Dict:
    """
    Returns
    -------

        dict(str: samna.device.DeviceInfo)
        Returns a dict of device name and device identifier.
    """
    def sort_devices(devices: List) -> List:
        devices.sort(key=lambda x: x.usb_device_address)
        return devices
    # Get all devices available
    devices = samna.device.get_all_devices()
    # Group by device_type_name
    device_groups = groupby(devices, lambda x: x.device_type_name)
    # Switch keys from samna's device_type_name to device_type names
    device_groups = {device_type_map[k]: sort_devices(list(v)) for k, v in device_groups}
    # Flat map
    for dev_type, dev_list in device_groups.items():
        for i, dev in enumerate(dev_list):
            device_map[f"{dev_type}:{i}"] = dev
    return device_map


def is_device_type(dev_info: samna.device.DeviceInfo, dev_type: str) -> bool:
    """
    Check if a DeviceInfo object is of a given device type `dev_type`

    Args
    ----

    dev_info: samna.device.DeviceInfo
        Device info object
    dev_type: str
        Device type as a string

    Returns:
    --------
        bool
    """
    return dev_info.device_type_name == device_types[dev_type]


def discover_device(device_id: str):
    """
    Discover a samna device by device_name:device_id pair

    Args
    ----

    device_id: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    Returns
    -------

    device_info: samna.device.DeviceInfo
    """
    device_info = device_map[device_id]
    return device_info


def open_device(device_id: str):
    """
    Open device function.

    Args
    ----

    device_id: str
        device_name:device_id pair given as a string

    Returns
    -------

    device_handle: samna.device.*
        Device handle received from samna.
    """
    device_map = get_device_map()
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    
    if device_handle is not None:
        return device_handle
    else:
        raise IOError("The connection to the device cannot be established.")


def close_device(device_id: str):
    """
    Close a device by device identifier

    Args
    ----

    device_id: str
        device_name:device_id pair given as a string.
        dynapcnndevkit:0 or speck:0 or dynapcnndevkit:1
    """
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    print(f"Closing device: {device_id}")
    samna.device.close_device(device_handle)


