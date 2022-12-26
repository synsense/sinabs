from sinabs.backend.dynapcnn.utils import parse_device_id, standardize_device_id


def test_device_id_no_index():
    assert parse_device_id("speck") == parse_device_id("speck:0")


def test_standardize():
    assert standardize_device_id("speck2b") == "speck2b:0"
    assert standardize_device_id("speck2b:00") == "speck2b:0"
    assert standardize_device_id("speck2b:1") == "speck2b:1"
