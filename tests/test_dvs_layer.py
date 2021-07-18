import torch
import torch.nn as nn


def test_init_defaults():
    from sinabs.backend.dynapcnn.dvslayer import DVSLayer

    dvs_layer = DVSLayer(input_shape=(128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


def test_construct():
    ...


def test_from_layers_empty():
    from sinabs.backend.dynapcnn.dvslayer import DVSLayer
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.flipdims import FlipDims

    dvs_layer = DVSLayer.from_layers(input_shape=(128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


def test_from_layers():
    from sinabs.backend.dynapcnn.dvslayer import DVSLayer
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.flipdims import FlipDims

    pool_layer = sl.SumPool2d(2)
    crop_layer = sl.Cropping2dLayer(((0, 5), (0, 10)))

    dvs_layer = DVSLayer.from_layers(input_shape=(128, 128), pool_layer=pool_layer, crop_layer=crop_layer)
    print(dvs_layer)

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = crop_layer(data)
    assert out.shape == (1, 2, (128 - 5), (128 - 10))

    out = dvs_layer(data)

    assert out.shape == (1, 2, (64 - 5), (64 - 10))

    assert dvs_layer.get_roi() == ((0, 59), (0, 54))
