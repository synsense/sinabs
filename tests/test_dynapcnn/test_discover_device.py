import pytest
import samna

from sinabs.backend.dynapcnn import io


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_list_all_devices():
    device_map = io.get_device_map()
    # Ideally the device map needs to be tested against something expected.
    raise NotImplementedError()


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_is_device_type():
    devices = samna.device.get_all_devices()
    print([io.is_device_type(d, "dynapcnndevkit") for d in devices])
    print([io.is_device_type(d, "dvxplorer") for d in devices])


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_get_device_map():
    device_map = io.get_device_map()
    print(device_map)
