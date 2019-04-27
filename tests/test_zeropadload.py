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
import pytest

strLibPath = sys.path[0]
sys.path.insert(1, strLibPath)


def test_loadkeras_modelZeroPadding():
    """
    Test the functionality of loading the model from a keras file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_model
    from tensorflow import keras

    # Initialize a Network
    torchModel = Network()
    strModel = strLibPath + "/models/testModel_zeropad.h5"
    try:
        with open(strModel) as _:
            # Do nothing
            raise FileNotFoundError
            pass
    except FileNotFoundError:
        createkeras_modelFile(strModel, padding="valid", zeropad=True)

    keras_model = keras.models.load_model(strModel)

    print(keras_model.to_json())
    torchModel = from_model(keras_model)

    # Verify no. of layers
    assert len(torchModel.layers) == len(keras_model.layers)


def test_inferDataFormat():
    """
    Test output of inferDataFormat from keras configuraiton
    :return:
    """
    from tensorflow import keras

    inputData = keras.layers.Input(shape=(20, 24, 3))
    x = keras.layers.Cropping2D(cropping=((0, 0), (2, 2)))(inputData)
    x = keras.layers.Conv2D(
        16,
        kernel_size=(4, 5),
        strides=(4, 5),
        padding="same",
        use_bias=False,
        activation="relu",
    )(x)
    x = keras.layers.Conv2D(
        16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        activation="relu",
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.ZeroPadding2D(padding=((1, 2), (3, 4)))(x)
    output = x
    model = keras.Model(inputs=[inputData], outputs=[output])

    from sinabs.from_keras import infer_data_format

    dataFormat = infer_data_format(model.get_config())
    assert dataFormat == "channels_last"


def createkeras_modelFile(strModel, padding="same", zeropad=False):
    """
    Create a model file for testing with a diverse set of layers
    """
    from tensorflow import keras

    inputData = keras.layers.Input(shape=(20, 24, 3))
    x = keras.layers.Cropping2D(cropping=((0, 0), (2, 2)))(inputData)
    x = keras.layers.Conv2D(
        16,
        kernel_size=(4, 5),
        strides=(4, 5),
        padding=padding,
        use_bias=False,
        activation="relu",
    )(x)
    x = keras.layers.Conv2D(
        16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding=padding,
        use_bias=False,
        activation="relu",
    )(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    if zeropad:
        x = keras.layers.ZeroPadding2D(padding=((1, 2), (3, 4)))(x)
    output = x
    model = keras.Model(inputs=[inputData], outputs=[output])
    model.save(strModel)
