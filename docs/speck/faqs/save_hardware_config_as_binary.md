# Save Samna Config As Binary

The hardware config can be saved in an external memory, and are used to configure the demo-kit on the fly.

To save the config as binary:

```python
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork

# init DynapcnnNetwork instance
dynapcnn = DynapcnnNetwork(snn=my_snn, input_shape=my_input_shape, dvs_input=True)
# make hardware config, suppose we're using Speck2fModuleDevkit
hardware_cfg = dynapcnn.make_config(device="speck2fmodule")

# usually, users need to use the "readout layer" on Speck
# so users must set the io_sel as 24(0b11000)
hardware_cfg.factory_config.io_sel = 24

# convert to binary
binary = samna.speck2f.configuration_to_flash_binary(hardware_cfg)
# save as binary
with open("./my_config.bin", "wb") as f:
    for e in binary:
        f.write((e).to_bytes(1, byteorder='little'))
```

More details can be found in [samna documetation.](https://synsense-sys-int.gitlab.io/samna/search.html?q=configuration_to_flash_binary)

