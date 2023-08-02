# Device Management

## How Do I List All Connected Devices And Their IDs?

Once your devices are connected, you can use the `get_device_map` method to inspect them.

### method1
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

### method2
```python
import samna

# Get device infos of all unopened devices
deviceInfos = samna.device.get_unopened_devices()

# Print device infos to see what devices are connected
for info in deviceInfos:
    print(info)
```

This should produce an output that looks something like below:

```
>>> device::DeviceInfo(serial_number=, usb_bus_number=2, usb_device_address=9, logic_version=0, device_type_name=Speck2fDevKit)
```

## How Do I Open The Device

```python
import samna
deviceInfos = samna.device.get_unopened_devices()
# Select the device you want to open, here we want to open the first one
device = samna.device.open_device(deviceInfos[0])
```

or you can just pass the devkit's name to the `open_device` function

```python
import samna
speck = samna.device.open_device("Speck2fModuleDevKit")
```

All devices that supported by samna can be found at [here.](https://synsense-sys-int.gitlab.io/samna/install.html#discover-supported-devices)

## How Do I Reset The Device

### method1(hard reset)
Just press the reset button on our devkit!

### method2(soft reset)
```python
import samna
# open device
speck = samna.device.open_device("Speck2fModuleDevKit")
...
...
...
# reset devkit, this operation will apply a default configuration to the devkit
speck.reset_board_soft(True)
```

## How Do I Close The Device
```python
import samna
# open device
speck = samna.device.open_device("Speck2fModuleDevKit")
...
...
...
# close device
samna.device.close_device(speck)
```
