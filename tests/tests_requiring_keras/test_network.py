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

def test_Network_dummyinit():
    from sinabs import Network

    # Test basic
    spkModel = Network()


def test_Network_init():
    from tensorflow import keras

    # Input layer
    imgDataFormat = "channels_first"
    inputs = keras.layers.Input(shape=(2, 50, 50))
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), data_format=imgDataFormat)(
        inputs
    )
    x = keras.layers.Conv2D(
        10, kernel_size=(3, 3), padding="same", data_format=imgDataFormat
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), data_format=imgDataFormat)(x)
    x = keras.layers.Conv2D(
        50, kernel_size=(3, 3), padding="same", data_format=imgDataFormat
    )(x)
    x = keras.layers.Flatten(data_format=imgDataFormat)(x)
    x = keras.layers.Dense(2)(x)

    keras_model = keras.Model(inputs=inputs, outputs=x)

    from sinabs import Network

    # Test initialize model
    spkModel = Network(analog_model=keras_model)


def test_Network_summary():
    from tensorflow import keras

    # Input layer
    imgDataFormat = "channels_first"
    inputs = keras.layers.Input(shape=(2, 50, 50))
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), data_format=imgDataFormat)(
        inputs
    )
    x = keras.layers.Conv2D(
        10, kernel_size=(3, 3), padding="same", data_format=imgDataFormat
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), data_format=imgDataFormat)(x)
    x = keras.layers.Conv2D(
        50, kernel_size=(3, 3), padding="same", data_format=imgDataFormat
    )(x)
    x = keras.layers.Flatten(data_format=imgDataFormat)(x)
    x = keras.layers.Dense(2)(x)

    keras_model = keras.Model(inputs=inputs, outputs=x)

    print(keras_model.summary())

    from sinabs import Network

    # Test initialize model
    spkModel = Network(analog_model=keras_model)

    print(spkModel.summary().to_string())
