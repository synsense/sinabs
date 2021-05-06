from sinabs.backend.dynapcnn import io
import pytest


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_list_all_devices():
    devices  = io.get_all_samna_devices()
    print(devices)
    for d in devices:
        print(d.to_json())
        print(d.device_type_name)


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_is_device_type():
    devices = io.get_all_samna_devices()
    print([io.is_device_type(d, "dynapcnndevkit") for d in devices])
    print([io.is_device_type(d, "dvxplorer") for d in devices])


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_discover_device():
    device = io.open_device("dvxplorer:0")
    device = io.open_device("dvxplorer:0")
    device = io.open_device("dynapcnndevkit:0")
    io.close_device("dvxplorer:0")
    io.close_device("dynapcnndevkit:0")
    assert(io.get_all_open_samna_devices() == [])


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_get_device_map():
    device_map = io.get_device_map()
    print(device_map)