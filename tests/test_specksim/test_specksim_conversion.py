import pytest


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,padding,stride,input_shape",
    [
        (2, 2, (3, 3), (1, 1), (1, 1), (2, 8, 8)),
        (2, 4, (7, 7), (3, 3), (1, 1), (4, 8, 8)),
    ],
)
def test_specksim_conv_layer_setup(
    in_channels, out_channels, kernel_size, padding, stride, input_shape
):
    import numpy as np
    from torch import nn

    from sinabs.backend.dynapcnn.specksim import convert_convolutional_layer

    input_shape = (2, 8, 8)
    layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=False,
    )
    conv_filter, _ = convert_convolutional_layer(layer, input_shape=input_shape)

    # test parameters
    assert in_channels == conv_filter.get_layer().get_in_channels()
    assert out_channels == conv_filter.get_layer().get_out_channels()
    assert kernel_size == conv_filter.get_layer().get_kernel_size()
    assert tuple(input_shape[1:]) == conv_filter.get_layer().get_in_shape()
    assert stride == conv_filter.get_layer().get_stride()
    assert padding == conv_filter.get_layer().get_padding()

    # test weight shape
    expected_weight_shape = (out_channels, in_channels, *kernel_size)
    assert expected_weight_shape == np.array(conv_filter.get_weights()).shape

    # test setting new weights
    new_weights = np.random.random(size=expected_weight_shape).astype(np.float32)
    conv_filter.set_weights(new_weights.tolist())
    assert (new_weights == np.array(conv_filter.get_weights())).all()


@pytest.mark.parametrize(
    "min_v_mem,spike_threshold,n_channels,input_shape",
    [(-1.0, 1.0, 2, (8, 8)), (0.0, 2.0, 4, (16, 16))],
)
def test_specksim_iaf_layer_setup(min_v_mem, spike_threshold, n_channels, input_shape):
    from sinabs.backend.dynapcnn.specksim import convert_iaf_layer
    from sinabs.layers import IAF

    iaf = IAF(min_v_mem=min_v_mem, spike_threshold=spike_threshold)
    iaf_filter, _ = convert_iaf_layer(layer=iaf, input_shape=(n_channels, *input_shape))

    # test initialization
    assert iaf_filter.get_layer().get_spike_threshold() == spike_threshold
    assert iaf_filter.get_layer().get_min_v_mem() == min_v_mem


def test_specksim_pool_layer_setup():
    from torch import nn

    from sinabs.backend.dynapcnn.specksim import convert_pooling_layer
    from sinabs.layers import SumPool2d

    kernel_size = (2, 2)
    stride = (2, 2)
    input_shape = (2, 8, 8)

    # convert from avg pool
    pool_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    pool_filter, _ = convert_pooling_layer(pool_layer, input_shape=input_shape)
    expected_output_shape = (
        input_shape[0],
        input_shape[1] // kernel_size[0],
        input_shape[2] // kernel_size[1],
    )

    assert pool_filter.get_layer().get_kernel_size() == kernel_size
    assert pool_filter.get_layer().get_shape() == tuple(input_shape[1:])
    assert pool_filter.get_layer().get_output_shape() == expected_output_shape[1:]

    # convert from avg pool
    pool_layer = SumPool2d(kernel_size=(2, 2), stride=(2, 2))
    pool_filter, _ = convert_pooling_layer(pool_layer, input_shape=input_shape)
    expected_output_shape = (
        input_shape[0],
        input_shape[1] // kernel_size[0],
        input_shape[2] // kernel_size[1],
    )

    assert pool_filter.get_layer().get_kernel_size() == kernel_size
    assert pool_filter.get_layer().get_shape() == tuple(input_shape[1:])
    assert pool_filter.get_layer().get_output_shape() == expected_output_shape[1:]
