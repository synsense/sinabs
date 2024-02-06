How to add support for a new chip
---------------------------------

In order to add support for a new chip in, you will need to perform the following steps.

1. Add the device to `device_types` dictionary in `io.py`.
   The dictionary comprises a `device_name_string` and its corresponding `device_name` in the samna `DeviceInfo` object.
   The `device_name_string` should be a word in lower case alphabets with no spaces or special characters. 
   The string should be a qualified variable name in python.
   The device is discovered based on the `DeviceInfor.device_name` and therefore should match the exact `device_name` string in `samna`.
   
2. Implement a `ConfigBuilder` for your device. 
   This builder describes how a model will be converted to a samna `config` object for your device type.
   You can base your implementation on the current implementations for other chips in the folder/module `sinabs.backend.dynapcnn.chips`
   
3. Add the `ConfigBuilder` implemented in step `2` to the `ChipFactory` in `chip_factory.py`.

That should be it! 
You will now be able to call `make_config` or the `to` method in `DynapcnnNetwork` 
and refer to your device with the name you chose `device_name_string` in step `1`.
