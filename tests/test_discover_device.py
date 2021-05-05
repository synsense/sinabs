from sinabs.backend.dynapcnn import io

def test_list_all_devices():
    devices  = io.get_all_samna_devices()
    print(devices)
    for d in devices:
        print(d.to_json())
        print(d.device_type_name)

def test_is_device_type():
    devices = io.get_all_samna_devices()
    print([io.is_device_type(d, "dynapcnndevkit") for d in devices])
    print([io.is_device_type(d, "dvxplorer") for d in devices])


def test_discover_device():
    device = io.open_device("dvxplorer:0")
    device = io.open_device("dynapcnndevkit:0")
    io.close_device("dvxplorer:0")
    io.close_device("dynapcnndevkit:0")
