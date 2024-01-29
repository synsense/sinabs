import math
import os
from itertools import groupby
from multiprocessing import Process
from typing import Dict, List, Tuple

import numpy as np
import samna
import samnagui
import torch

from .utils import parse_device_id, standardize_device_id

# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck": "speck",
    "speck2b": "Speck2bTestboard",
    "speck2devkit": "Speck2DevKit",
    "speck2btiny": "Speck2bDevKitTiny",
    "speck2e": "Speck2eTestBoard",  # with a capital B for board
    "speck2edevkit": "Speck2eDevKit",
    "speck2fmodule": "Speck2fModuleDevKit",
    "speck2fdevkit": "Speck2fDevKit",
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
    """Disable timestamps of the samna node.

    Args
    ----
    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_id = standardize_device_id(device_id=device_id)
    device_name, device_idx = parse_device_id(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 1)
    else:
        device_handle.get_stop_watch().set_enable_value(True)


def disable_timestamps(
    device_id: str,
) -> None:
    """Disable timestamps of the samna node.

    Args
    ----

    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_id = standardize_device_id(device_id=device_id)
    device_name, device_idx = parse_device_id(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 0)
    else:
        device_handle.get_stop_watch().set_enable_value(False)


def reset_timestamps(
    device_id: str,
) -> None:
    """Disable timestamps of the samna node.

    Args
    ----
    device_id: str
        Name of the device to initialize. Required for different existing APIs
        for Dynapcnndevkit and Speck chips
    """
    device_id = standardize_device_id(device_id=device_id)
    device_name, device_idx = parse_device_id(device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    if device_name.lower() == "dynapcnndevkit":
        device_handle.get_io_module().write_config(0x0003, 1)
    else:
        device_handle.get_stop_watch().reset()


# def events_to_raster(event_list: List, layer: int) -> torch.Tensor:
#    """
#    Convert an eventList read from `samna` to a tensor `raster` by filtering only the events specified by `layer`.
#
#    Parameters
#    ----------
#
#    event_list: List
#        A list comprising of events from samna API
#
#    layer: int
#        The index of layer for which the data needs to be converted
#
#    Returns
#    -------
#
#    raster: torch.Tensor
#    """
#    evs_filtered = filter(
#        lambda x: isinstance(x, samna.dynapcnn.event.Spike), event_list
#    )
#    evs_filtered = filter(lambda x: x.layer == layer, evs_filtered)
#    raise NotImplementedError
#    raster = map(rasterize, evs_filtered)
#    return raster


def events_to_xytp(event_list: List, layer: int) -> np.array:
    """Convert an eventList read from `samna` to a numpy structured array of `x`, `y`, `t`,
    `channel`.

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
    evs_filtered = list(
        filter(
            lambda x: isinstance(x, samna.dynapcnn.event.Spike) and x.layer == layer,
            event_list,
        )
    )
    xytc = np.empty(
        len(evs_filtered),
        dtype=[
            ("x", np.uint16),
            ("y", np.uint16),
            ("t", np.uint64),
            ("channel", np.uint16),
        ],
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
    device_groups = {
        device_type_map[k]: sort_devices(list(v)) for k, v in device_groups
    }
    # Flat map
    for dev_type, dev_list in device_groups.items():
        for i, dev in enumerate(dev_list):
            device_map[f"{dev_type}:{i}"] = dev
    return device_map


def is_device_type(dev_info: samna.device.DeviceInfo, dev_type: str) -> bool:
    """Check if a DeviceInfo object is of a given device type `dev_type`

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
    """Discover a samna device by device_name:device_id pair.

    Args
    ----

    device_id: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    Returns
    -------

    device_info: samna.device.DeviceInfo
    """
    device_id = standardize_device_id(device_id=device_id)
    device_info = device_map[device_id]
    return device_info


def open_device(device_id: str):
    """Open device function.

    Args
    ----

    device_id: str
        device_name:device_id pair given as a string

    Returns
    -------

    device_handle: samna.device.*
        Device handle received from samna.
    """
    device_id = standardize_device_id(device_id=device_id)
    device_map = get_device_map()
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)

    if device_handle is not None:
        return device_handle
    else:
        raise IOError("The connection to the device cannot be established.")


def close_device(device_id: str):
    """Close a device by device identifier.

    Args
    ----

    device_id: str
        device_name:device_id pair given as a string.
        dynapcnndevkit:0 or speck:0 or dynapcnndevkit:1
    """
    device_id = standardize_device_id(device_id=device_id)
    device_info = device_map[device_id]
    device_handle = samna.device.open_device(device_info)
    print(f"Closing device: {device_id}")
    samna.device.close_device(device_handle)


def launch_visualizer(
    receiver_endpoint: str,
    width_proportion: float = 0.6,
    height_proportion: float = 0.6,
    disjoint_process: bool = True,
):
    """Launch the samna visualizer in a separate process.

    NOTE: MacOS users will want to use disjoint_process as True as a GUI process cannot be launched as a subprocess.

    Args:
        receiver_endpoint (str): the visualiser’s endpoint for receiving events (e.g. “tcp://0.0.0.0:33335”).
        width_proportion (bool): the rate between window width and workarea width of main monitor, default 0.75 which means this window has a width which equals to 3/4 width of main monitor’s workarea.
        height_proportion (bool): the rate between window height and workarea height of main monitor, default 0.75 which means this window has a height which equals to 3/4 height of main monitor’s workarea.
        disjoint_process (bool, optional): If true, will be launched in a disjoint shell process. Defaults to True. If false, this just runs the default samna command.

    Returns:
        gui_process (Process): The gui sub-process handle if disjoint_process was False.
    """
    if disjoint_process:
        os.system(
            f"samnagui -W {width_proportion} -H {height_proportion} {receiver_endpoint} &"
        )
    else:
        gui_process = Process(
            target=samnagui.run_visualizer,
            args=(receiver_endpoint, width_proportion, height_proportion),
        )
        gui_process.start()
        return gui_process


def calculate_neuron_address(
    x: int, y: int, c: int, feature_map_size: Tuple[int, int, int]
) -> int:
    """Calculate the neuron address on the devkit. This function is designed for ReadNeuronValue
    event to help the user check the neuron value of the SNN on the devkit.

    Args
    ----

    x: int
        x coordinate of the neuron
    y: int
        y coordinate of the neuron
    c: int
        channel index of the neuron
    feature_map_size: Tuple[int, int, int]
        the size of the feature map [channel, height, width]

    Returns
    ----

    neuron_address: int
    """
    # calculate how many bits it takes based on the feature map size
    channel, height, width = feature_map_size
    x_bits = math.ceil(math.log2(width))
    y_bits = math.ceil(math.log2(height))
    channel_bits = math.ceil(math.log2(channel))
    assert (
        x_bits + y_bits + channel_bits <= 18
    ), "Bits overflow! Check if your input arguments are correct!"

    x_shift_bits = channel_bits
    y_shift_bits = channel_bits + y_bits
    y_address = y << y_shift_bits
    x_address = x << x_shift_bits
    c_address = c

    neuron_address = y_address | x_address | c_address

    return neuron_address


def neuron_address_to_cxy(
    address: int, feature_map_size: Tuple[int, int, int]
) -> Tuple:
    """Calculate the c, x, y, coordinate of a neuron when the address of the NeuronValue event is
    given.

    Args
    ----

    address: int
        the neuron address of the NeuronValue event
    feature_map_size: Tuple[int, int, int]
        the size of the feature map [channel, height, width]

    Returns
    ----

    neuron_cxy: Tuple[int, int, int]
        the [channel, x, y] of the neuron
    """
    # calculate how many bits it takes based on the feature map size
    channel, height, width = feature_map_size
    x_bits = math.ceil(math.log2(width))
    y_bits = math.ceil(math.log2(height))
    channel_bits = math.ceil(math.log2(channel))
    assert (
        x_bits + y_bits + channel_bits <= 18
    ), "Bits overflow! Check if your input arguments are correct!"

    x_shift_bits = channel_bits
    y_shift_bits = channel_bits + y_bits

    y = address >> y_shift_bits
    x = (address >> x_shift_bits) & (2**x_bits - 1)
    c = address & (2**channel_bits - 1)

    return c, x, y
