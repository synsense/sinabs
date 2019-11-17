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

import sys

#
strLibPath = sys.path[0] + "/../"
sys.path.insert(1, strLibPath)


#def test_compare_keras_spiking():
#    import torch  # Torch must be imported after tensorflow to avoid a crash
#
#    from sinabs.network import Network
#    from sinabs.from_keras import from_model
#
#    # Initialize a Network
#    torchModel = Network()
#
#    # Load face detection model
#    keras_model = createkeras_model()
#    from_model(keras_model, network=torchModel)
#
#    ## Create input
#    modelInput = (torch.rand((10, 1, 260, 346)) > 0.94).float()
#
#    # Compare activity of analog vs spiking
#    act, rates = torchModel.compare_activations(modelInput, with_keras=True)
#
#    assert len(act) == len(rates)
#
#    # Compare shapes of Keras activations to the converted model
#    for idx in range(len(act)):
#        print(act[idx].shape, rates[idx].shape)
#        assert act[idx].shape == rates[idx].shape


def test_compare_analog_spiking():
    import torch  # Torch must be imported after tensorflow to avoid a crash

    from sinabs.network import Network
    from sinabs.from_keras.from_keras import from_model

    # Initialize a Network
    torchModel = Network()

    keras_model = createkeras_model()
    from_model(keras_model, network=torchModel)
    print(keras_model.summary())
    print(torchModel.summary().to_string())

    # Create input
    tsrInp = (torch.rand((10, 1, 260, 346)) > 0.94).float()

    # Compare activity of analog vs spiking models
    act, rates = torchModel.compare_activations(tsrInp, with_keras=False)

    assert len(act) == len(rates)

    # Compare shapes of Keras activations to the converted model
    for idx in range(len(act)):
        assert act[idx].shape[1:] == rates[idx].shape


def createkeras_model():
    """
    Create a keras model
    :return:
    """
    from tensorflow import keras

    kernel_size = (3, 3)
    pool_size = (2, 2)
    strides = (2, 2)
    activation = "relu"
    bias = False

    inputLayer = keras.Input((1, 260, 346))

    x = keras.layers.Cropping2D(cropping=(2, 45), data_format="channels_first")(
        inputLayer
    )

    x = keras.layers.AveragePooling2D(
        pool_size=(4, 4), strides=(4, 4), data_format="channels_first", padding="valid"
    )(x)

    x = keras.layers.ZeroPadding2D(
        padding=((1, 0), (1, 0)), data_format="channels_first"
    )(x)
    x = keras.layers.Conv2D(
        16,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        data_format="channels_first",
        activation=activation,
        use_bias=bias,
    )(x)
    x = keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=pool_size,
        data_format="channels_first",
        padding="valid",
    )(x)

    x = keras.layers.ZeroPadding2D(
        padding=((1, 0), (1, 0)), data_format="channels_first"
    )(x)
    x = keras.layers.Conv2D(
        32,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        data_format="channels_first",
        activation=activation,
        use_bias=bias,
    )(x)
    x = keras.layers.AveragePooling2D(
        pool_size=pool_size,
        strides=pool_size,
        data_format="channels_first",
        padding="valid",
    )(x)

    x = keras.layers.ZeroPadding2D(
        padding=((1, 0), (1, 0)), data_format="channels_first"
    )(x)
    x = keras.layers.Conv2D(
        64,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        data_format="channels_first",
        activation=activation,
        use_bias=bias,
    )(x)
    x = keras.layers.Conv2D(
        128,
        kernel_size=(2, 2),
        strides=(2, 2),
        data_format="channels_first",
        padding="valid",
        activation=activation,
        use_bias=bias,
    )(x)

    x = keras.layers.Conv2D(
        2,
        kernel_size=(1, 1),
        strides=(1, 1),
        data_format="channels_first",
        padding="valid",
        activation=activation,
        # use_bias=True,
    )(x)

    keras_model = keras.Model(inputLayer, x)
    return keras_model
