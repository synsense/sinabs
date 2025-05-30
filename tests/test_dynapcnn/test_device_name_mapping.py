from sinabs.backend.dynapcnn.utils import parse_device_id, standardize_device_id


def test_device_id_no_index():
    assert parse_device_id("speck") == parse_device_id("speck:0")


# TODO: evaluate if we can use another board for this test instead of speck 2b
# def test_standardize():
#     assert standardize_device_id("") == ":0"
#     assert standardize_device_id(":00") == ":0"
#     assert standardize_device_id(":1") == ":1"
