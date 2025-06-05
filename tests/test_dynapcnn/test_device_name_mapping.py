from sinabs.backend.dynapcnn.utils import parse_device_id, standardize_device_id


def test_device_id_no_index():
    assert parse_device_id("speck") == parse_device_id("speck:0")


def test_standardize():
    assert standardize_device_id("speck2f") == "speck2f:0"
    assert standardize_device_id("speck2f:00") == "speck2f:0"
    assert standardize_device_id("speck2f:1") == "speck2f:1"
