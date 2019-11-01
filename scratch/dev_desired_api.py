
devices = samna.discoverer.get_devices()
for device_info in devices:
    if device_info.something == "WhatIWant":
        device_model = samna.discoverer.get_device_model(device)

device_model.set_config(config: samna.SpeckConfig)
device_model.set_weights(my_weights)
device_model.apply()

device_model.send_events(some_events)
out = device_model.get_events()