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

strLibPath = sys.path[0]
sys.path.insert(1, strLibPath)


def test_loadFromJson():
    """
    Load a model from json file
    """
    createkeras_model()
    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".json"

    from sinabs.from_keras import from_model_keras_config
    import json

    with open(filename) as fp:
        keras_config = json.load(fp)

    torchAngModel = from_model_keras_config(keras_config=keras_config)

    print(torchAngModel)


def test_loadFromJsonSpiking():
    """
    Load a spiking model from json file
    """
    createkeras_model()
    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".json"

    from sinabs.from_keras import from_model_keras_config
    import json

    with open(filename) as fp:
        keras_config = json.load(fp)

    torchNetwork = from_model_keras_config(keras_config=keras_config, spiking=True)
    print(torchNetwork)


def test_instantiate_sinabs_from_Json():
    """
    Test the functionality of loading the model from a json file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_json

    createkeras_model()
    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".json"

    # Initialize a Network
    torchModel = Network()

    with open(filename) as f:
        for l in f.readlines():
            keras_json = l

    from_json(keras_json=keras_json, input_shape=None, network=torchModel)

    print(torchModel.summary())


def test_set_weights_fromNumpy_DenseLayers():
    """
    Test loading weights from a numpy pickle file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_json
    import numpy as np

    createkeras_model()
    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".json"

    # Initialize a Network
    torchModel = Network()

    with open(filename) as f:
        for l in f.readlines():
            keras_json = l

    from_json(keras_json=keras_json, input_shape=None, network=torchModel)

    weightfile = strLibPath + "/weights/" + modelname + ".npy"

    torchModel.set_weights(np.load(weightfile), is_from_keras=True)


def test_set_weights_fromNumpy():
    """
    Test loading weights from a numpy pickle file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_json
    import numpy as np

    # Initialize a Network
    torchModel = Network()
    createkeras_model()
    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".json"

    with open(filename) as f:
        for l in f.readlines():
            keras_json = l

    from_json(keras_json=keras_json, input_shape=None, network=torchModel)

    weightfile = strLibPath + "/weights/" + modelname + ".npy"

    torchModel.set_weights(np.load(weightfile), is_from_keras=True)
    print(torchModel.summary().to_string())


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

    modelname = "testModelWithJson"

    filename = strLibPath + "/models/" + modelname + ".h5"
    keras_model.save(filename)
    import json

    filename = strLibPath + "/models/" + modelname + ".json"
    with open(filename, "w") as fp:
        fp.write(keras_model.to_json())
    filename = strLibPath + "/weights/" + modelname
    import numpy as np

    np.save(filename, keras_model.get_weights())
