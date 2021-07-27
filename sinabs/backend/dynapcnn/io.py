import samna
import torch
from itertools import groupby
from typing import List, Dict
import numpy as np

# Managed global variables
samna_node = None
samna_devices = None

# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck": "speck",
    "speck2b": "Speck2bTestboard",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
    "speck2devkit": "Speck2DevKit",
    "dynapse1devkit": "Dynapse1DevKit",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
    "pollendevkit": "PollenDevKit",
    "dynapcnndevkit": "DynapcnnDevKit",
}

device_type_map = {v: k for (k, v) in device_types.items()}


def enable_timestamps(device: str) -> None:
    """
    Enable timestamping of events

    Parameters
    ----------
    device: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    """
    dev_name, _ = _parse_device_string(device)
    if dev_name == "dynapcnndevkit":
        device = open_device(device)
        device.get_io_module().write_config(0x0003, 1)
    else:
        device = open_device(device)
        stopWatch = device.get_stop_watch()
        stopWatch.set_enable_value(True) # to enable


def disable_timestamps(device: str) -> None:
    """
    Disable timestamping of events

    Parameters
    ----------
    device: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    """
    dev_name, _ = _parse_device_string(device)
    if dev_name == "dynapcnndevkit":
        device = open_device(device)
        device.get_io_module().write_config(0x0003, 0)
    else:
        device = open_device(device)
        stopWatch = device.get_stop_watch()
        stopWatch.set_enable_value(False) # to enable


def reset_timestamps(device: str) -> None:
    """
    Reset the timeer to 0

    Parameters
    ----------
    device: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    """
    dev_name, _ = _parse_device_string(device)
    if dev_name == "dynapcnndevkit":
        disable_timestamps(device)
    else:
        device = open_device(device)
        stopWatch = device.get_stop_watch()
        stopWatch.reset() # to reset to 0 (it doesn't disable it automatically, so it will go on coutning)




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


def init_samna_node():
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
    return samna_node


def get_samna_node():
    global samna_node

    if samna_node is None:
        # Initialize the node
        samna_node = init_samna_node()
        # Fetch all samna devices
        get_all_samna_devices()

    return samna_node


def get_all_unopened_samna_devices():
    """
    Returns a list of all unopen samna devices

    Returns
    -------
    devices:
        A list of samna devices currently un-opened
    """

    get_samna_node()
    # Find device
    return samna.device_node.DeviceController.get_unopened_devices()


def get_all_open_samna_devices():
    """
    Returns a list of all opened samna devices

    Returns
    -------
    devices:
        A list of samna devices currently opened

    """
    get_samna_node()
    # Find device
    return [x.device_info for x in samna.device_node.DeviceController.get_opened_devices()]


def get_all_samna_devices():
    """
    Returns
    -------
    devices:
        Returns a list of all available samna devices
    """
    global samna_devices
    if samna_devices is None:
        samna_devices = get_all_open_samna_devices() + get_all_unopened_samna_devices()
    return samna_devices


def get_device_map() -> Dict:
    """
    Returns
    -------
    Returns a dict of device name and device info
    """

    def sort_devices(devices: List) -> List:
        devices.sort(key=lambda x: x.usb_device_address)
        return devices

    # Get all devices available
    devices = get_all_samna_devices()
    # Group by device_type_name
    device_groups = groupby(devices, lambda x: x.device_type_name)
    # Switch keys from samna's device_type_name to device_type names
    device_groups = {device_type_map[k]: sort_devices(list(v)) for k, v in device_groups}
    # Flat map
    device_map = {}
    for dev_type, dev_list in device_groups.items():
        for i, dev in enumerate(dev_list):
            device_map[f"{dev_type}:{i}"] = dev
    return device_map


def is_device_type(dev_info: samna.device.DeviceInfo, dev_type: str) -> bool:
    """
    Check if a DeviceInfo object is of a given device type `dev_type`

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
    return dev_info.device_type_name == device_types[dev_type]


def discover_device(device_id: str):
    """
    Discover a samna device

    Parameters
    ----------
    device_id: str
        Device name/identifier (dynapcnndevkit:0 or speck:0 or dvxplorer:1 ... )
        The convention is similar to that of pytorch GPU identifier ie cuda:0 , cuda:1 etc.

    Returns
    -------
    device_info: samna.device.DeviceInfo

    See Also
    --------
    open_device, close_device

    """

    device_map = get_device_map()
    device_info = device_map[device_id]
    return device_info


def open_device(device_id: str):
    """

    Parameters
    ----------
    device_id

    Returns
    -------

    """
    device_name, device_num = _parse_device_string(device_id)
    device_id = f"{device_name}:{device_num}"
    device_map = get_device_map()
    dev_info = device_map[device_id]
    name = f"{device_name}_{device_num}"
    if name in samna.device_node.__dict__:
        return samna.device_node.__dict__[name]
    else:
        # Open Devkit
        samna.device_node.DeviceController.open_device(dev_info, name)
        # get the handle of our dev-kit
        device = samna.device_node.__dict__[name]
        return device


def close_device(device_id: str):
    """
    Convenience method to close a device

    Parameters
    ----------
    device_id: str
        Device name/identifier (DynaapcnnDevKit:0 or speck:0 or DynaapcnnDevKit:1 ... )

    """
    device_name, device_num = _parse_device_string(device_id)
    samna.device_node.DeviceController.close_device(f"{device_name}_{device_num}")


def _parse_device_string(device_id: str) -> (str, int):
    device_splits = device_id.split(":")
    device_name = device_splits[0]
    if len(device_splits) > 1:
        device_num = int(device_splits[1])
    else:
        device_num = 0
    return device_name, device_num
