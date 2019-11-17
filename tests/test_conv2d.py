#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

def test_import():
    """
    Basic syntax check test
    """


def test_TorchSpikingConv2dLayer_initialization():
    """
    Test initialization of Conv2D layer
    :return:
    """
    from sinabs.layers import SpikingConv2dLayer

    lyr = SpikingConv2dLayer(
        channels_in=2,
        image_shape=(10, 10),
        channels_out=6,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 0, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="conv2d",
    )

    assert lyr.output_shape == (6, 8, 10)


def test_getoutput_shape():
    from sinabs.layers import SpikingConv2dLayer
    import torch

    lyr = SpikingConv2dLayer(
        channels_in=2,
        image_shape=(10, 20),
        channels_out=6,
        kernel_shape=(3, 5),
        strides=(1, 1),
        padding=(0, 3, 6, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="conv2d",
    )

    tsrInput = (torch.rand(10, 2, 10, 20) > 0.9).float()

    tsrOutput = lyr(tsrInput)
    assert lyr.output_shape == tsrOutput.shape[1:]


def test_summary():
    from sinabs.layers import SpikingConv2dLayer

    lyr = SpikingConv2dLayer(
        channels_in=2,
        image_shape=(10, 20),
        channels_out=6,
        kernel_shape=(3, 5),
        strides=(1, 1),
        padding=(0, 3, 6, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="conv2d",
    )

    # Runtime test to verify summary generation test
    print(lyr.summary())


def test_createConv2dFromKeras():
    from tensorflow import keras

    kerasLayer = keras.layers.Conv2D(
        3,
        kernel_size=(5, 3),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
    )
    keras_config = kerasLayer.get_config()

    from sinabs.from_keras.from_keras import from_conv2d_keras_conf

    # Create spiking layers
    layer_list = from_conv2d_keras_conf(
        keras_config,
        input_shape=(5, 30, 50),
        spiking=True,
        quantize_analog_activation=True,
    )
    for layer_name, layer in layer_list:
        print(layer.summary())
        output_shape = layer.output_shape

    # Create non spiking layers
    layer_list = from_conv2d_keras_conf(
        keras_config,
        input_shape=(5, 30, 50),
        spiking=False,
        quantize_analog_activation=False,
    )
    for layer_name, layer in layer_list:
        print(layer_name)

    assert output_shape == (3, 30, 50)


def test_createLayerFromConf():
    from tensorflow import keras

    kerasLayer = keras.layers.Conv2D(
        3,
        kernel_size=(5, 3),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
    )
    keras_model = keras.Sequential([kerasLayer])

    keras_config = keras_model.get_config()

    from sinabs.from_keras.from_keras import from_layer_keras_conf

    # Create non spiking layers
    layer_list = from_layer_keras_conf(
        keras_config["layers"][0],
        input_shape=(5, 30, 50),
        spiking=False,
        quantize_analog_activation=False,
    )
    for layer_name, layer in layer_list:
        print(layer_name)

    # Create spiking layers
    layer_list = from_layer_keras_conf(
        keras_config["layers"][0],
        input_shape=(5, 30, 50),
        spiking=True,
        quantize_analog_activation=True,
    )
    for layer_name, layer in layer_list:
        print(layer.summary())
        output_shape = layer.output_shape

    assert output_shape == (3, 30, 50)


def test_denseLayerFromKeras_spiking():
    from tensorflow import keras

    kerasLayer = keras.layers.Dense(30, activation="relu", use_bias=True)
    keras_config = kerasLayer.get_config()

    from sinabs.from_keras.from_keras import from_dense_keras_conf

    layer_list = from_dense_keras_conf(
        keras_config, input_shape=(3, 8, 8), spiking=True
    )

    for layer_name, layer in layer_list:
        print(layer_name)
        print(layer.summary())

    assert layer.output_shape == (30, 1, 1)
    assert layer.kernel_shape == (8, 8)


def test_denseLayerFromKeras_non_spiking():
    from tensorflow import keras

    kerasLayer = keras.layers.Dense(30, activation="relu", use_bias=True)
    keras_config = kerasLayer.get_config()

    from sinabs.from_keras.from_keras import from_dense_keras_conf

    layer_list = from_dense_keras_conf(keras_config, input_shape=(50,), spiking=False)

    for layer_name, layer in layer_list:
        print(layer_name)

    # TODO : Verify dimensions of output ?


def test_TorchSpikingConv2dLayer_forward():
    """
    Test forward pass of Conv2D layer has desired behaviour
    :return:
    """
    from sinabs.layers import SpikingConv2dLayer
    import torch
    from numpy import allclose

    lyr = SpikingConv2dLayer(
        channels_in=1,
        image_shape=(2, 2),
        channels_out=1,
        kernel_shape=(1, 1),
        strides=(1, 1),
        padding=(0, 0, 0, 0),
        bias=False,
        threshold=2.2,
        threshold_low=-2.2,
        membrane_subtract=2.2,
        layer_name="conv2d",
    )

    with torch.no_grad():

        weights = torch.Tensor([[[[1.0]]]])
        input_signal = torch.Tensor([[[[1., -2.], [-6., 8.]]],
                                     [[[1., -2.], [-6., 8.]]],
                                     [[[1., -2.], [-6., 8.]]]])
        expected_out = torch.Tensor([[[[0, 0], [0, 3]]],
                                     [[[0, 0], [0, 4]]],
                                     [[[1, 0], [0, 3]]]])
        expected_final_state = torch.Tensor([[[.8, -2.2], [-2.2, 2.0]]])

        lyr.conv.weight.data = weights

        out = lyr(input_signal)
        final_state = lyr.state

    assert allclose(out, expected_out)
    assert allclose(final_state, expected_final_state)


def test_TorchSpikingConv2dLayer_linear():
    """
    Test forward pass of Conv2D layer has desired behaviour
    :return:
    """
    from sinabs.layers import SpikingConv2dLayer
    import torch
    from numpy import allclose

    lyr = SpikingConv2dLayer(
        channels_in=1,
        image_shape=(2, 2),
        channels_out=1,
        kernel_shape=(1, 1),
        strides=(1, 1),
        padding=(0, 0, 0, 0),
        bias=False,
        threshold=2.2,
        threshold_low=-2.2,
        membrane_subtract=2.2,
        layer_name="conv2d",
        negative_spikes=True
    )

    with torch.no_grad():

        weights = torch.Tensor([[[[1.0]]]])
        input_signal = torch.Tensor([[[[1., -2.], [-6., 8.]]],
                                     [[[1., -2.], [-6., 8.]]],
                                     [[[1., -2.], [-6., 8.]]]])
        expected_out = torch.Tensor([[[[0, 0], [-2, 3]]],
                                     [[[0, -1], [-3, 4]]],
                                     [[[1, -1], [-3, 3]]]])
        expected_final_state = torch.Tensor([[[.8, -1.6], [-0.4, 2.0]]])

        lyr.conv.weight.data = weights

        out = lyr(input_signal)
        final_state = lyr.state

    assert allclose(out, expected_out)
    assert allclose(final_state, expected_final_state)