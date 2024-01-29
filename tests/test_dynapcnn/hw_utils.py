import torch

import sinabs
import sinabs.backend.dynapcnn as sdl


def get_network(has_identity_weights=True, n_layers=9, has_bias=False):
    sequential = torch.nn.Sequential()
    for i in range(n_layers):
        layer = torch.nn.Conv2d(1, 1, (1, 1), bias=has_bias)
        if has_identity_weights:
            with torch.no_grad():
                layer.weight *= 0
                layer.weight += 1
        sequential.add_module(f"{i * 3}", layer)
        sequential.add_module(f"{i * 3 + 1}", torch.nn.ReLU())
        sequential.add_module(
            f"{i * 3 + 2}", torch.nn.AvgPool2d(kernel_size=1, stride=1)
        )
    return sequential


def get_ones_network():
    snn = sinabs.from_torch.from_model(
        get_network(True, 1, False), input_shape=(1, 1, 1), batch_size=1
    ).spiking_model
    dynapcnn_network = sdl.DynapcnnNetwork(snn, input_shape=(1, 1, 1), dvs_input=False)
    return dynapcnn_network


supported_device_types_for_testing = {
    "speck": "speck",
    "speck2b": "Speck2bTestboard",
    "speck2devkit": "Speck2DevKit",
    "speck2btiny": "Speck2bDevKitTiny",
    "speck2e": "Speck2eTestBoard",
    "speck2edevkit": "Speck2eDevKit",
    "speck2fmodule": "Speck2fModuleDevKit",
    "dynapcnndevkit": "DynapcnnDevKit",
}


def find_open_devices():
    import samna

    reverse_dict = {v: k for k, v in supported_device_types_for_testing.items()}
    dev_infos = samna.device.get_all_devices()
    dev_dict = {}
    for dev_info in dev_infos:
        dev_dict.update({reverse_dict[dev_info.device_type_name]: dev_info})
    return dev_dict


def is_any_samna_device_connected():
    return len(find_open_devices()) > 0


def is_device_connected(device_type: str):
    return device_type in find_open_devices().keys()


def reset_all_connected_boards():
    print("Boards are being reset!")
    # this step is necessary as the gitlab-ci runner may send the chip erroneous events
    import samna

    devs = samna.device.get_unopened_devices()
    if len(devs) > 0:  # check if the connected board is found.
        for device in devs:
            handle = samna.device.open_device(device)
            handle.reset_board_soft(True)
            samna.device.close_device(handle)
            print(f"Resetted board: {device.device_type_name}")
