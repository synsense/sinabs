# Device Management

## How do I list all connected devices and their IDs?

Once your devices are connected, you can use the `get_device_map` method to inspect them.

```python
from sinabs.backend.dynapcnn.io import get_device_map
from typing import Dict

device_map: Dict[str, 'DeviceInfo'] = get_device_map()

print(device_map)
```

This should produce an output that looks something like below:

```
>>> {'speck2edevkit:0': device::DeviceInfo(serial_number=, usb_bus_number=0, usb_device_address=5, logic_version=0, device_type_name=Speck2eDevKit)}
```