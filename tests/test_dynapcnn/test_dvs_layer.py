from itertools import product

import pytest
import torch
import torch.nn as nn


def test_init_defaults():
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer

    dvs_layer = DVSLayer(input_shape=(128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


def test_from_layers_empty():
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
    from sinabs.backend.dynapcnn.flipdims import FlipDims

    dvs_layer = DVSLayer.from_layers(input_shape=(2, 128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


params = tuple(product((True, False), (0, 1, 2, 3)))


@pytest.mark.parametrize("disable_pixel_array,num_channels", params)
def test_from_layers(disable_pixel_array, num_channels):
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.crop2d import Crop2d
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer

    pool_layer = sl.SumPool2d(2)
    crop_layer = Crop2d(((0, 59), (0, 54)))

    kwargs_layer = dict(
        input_shape=(num_channels, 128, 128),
        pool_layer=pool_layer,
        crop_layer=crop_layer,
        disable_pixel_array=disable_pixel_array,
    )

    if 0 < num_channels <= 2:
        dvs_layer = DVSLayer.from_layers(**kwargs_layer)
    else:
        with pytest.raises(ValueError):
            dvs_layer = DVSLayer.from_layers(**kwargs_layer)
        return

    print(dvs_layer)

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = crop_layer(data)
    assert out.shape == (1, 2, 59, 54)

    out = dvs_layer(data)

    assert out.shape == (1, num_channels, (64 - 5), (64 - 10))

    assert dvs_layer.get_roi() == ((0, 59), (0, 54))


def test_convert_cropping2dlayer_to_crop2d():
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.utils import convert_cropping2dlayer_to_crop2d

    input_shape = (64, 50)
    cropping_lyr = sl.Cropping2dLayer(((1, 4), (3, 7)))
    crop2d_lyr = convert_cropping2dlayer_to_crop2d(cropping_lyr, input_shape)

    assert crop2d_lyr.top_crop == cropping_lyr.top_crop
    assert crop2d_lyr.left_crop == cropping_lyr.left_crop
    assert crop2d_lyr.bottom_crop == 64 - 4
    assert crop2d_lyr.right_crop == 50 - 7
