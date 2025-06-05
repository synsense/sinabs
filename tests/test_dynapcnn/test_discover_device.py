import pytest
import samna

from sinabs.backend.dynapcnn import io
from hw_utils import find_open_devices


@pytest.mark.skip("Not implemented")
def test_list_all_devices():
    devices = find_open_devices()

    if len(devices) == 0:
        pytest.skip("A connected Speck device is required to run this test")

    device_map = io.get_device_map()
    # Ideally the device map needs to be tested against something expected.
    raise NotImplementedError()


def test_is_device_type():
    devices = find_open_devices()

    if len(devices) == 0:
        pytest.skip("A connected Speck device is required to run this test")

    devices = samna.device.get_all_devices()
    print([io.is_device_type(d, "speck2fdevkit") for d in devices])
    print([io.is_device_type(d, "dvxplorer") for d in devices])


def test_get_device_map():

    devices = find_open_devices()

    if len(devices) == 0:
        pytest.skip("A connected Speck device is required to run this test")

    device_map = io.get_device_map()
    print(device_map)
