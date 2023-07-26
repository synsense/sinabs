# Device Management

## How do I list all connected devices and their IDs?

Once your devices are connected, you can use the `get_device_map` method to inspect them.

## method1
```python
from sinabs.backend.dynapcnn.io import get_device_map
from typing import Dict

device_map: Dict[str, 'DeviceInfo'] = get_device_map()

print(device_map)
```

This should produce an output that looks something like below:

```
>>> {'speck2fdevkit:0': device::DeviceInfo(serial_number=, usb_bus_number=2, usb_device_address=9, logic_version=0, device_type_name=Speck2fDevKit)}
```

## method2
```python
import samna

# Get device infos of all unopened devices
deviceInfos = samna.device.get_unopened_devices()

# Print device infos to see what devices are connected
for info in deviceInfos:
    print(info)

# Select the device you want to open, here we want to open the first one
device = samna.device.open_device(deviceInfos[0])
```

This should produce an output that looks something like below:

```
>>> device::DeviceInfo(serial_number=, usb_bus_number=2, usb_device_address=9, logic_version=0, device_type_name=Speck2fDevKit)
```