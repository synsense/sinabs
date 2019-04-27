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


def test_loadkeras_model():
    """
    Test the functionality of loading the model from a keras file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_model

    # Initialize a Network
    torchModel = Network()
    strModel = strLibPath + "/models/testModel.h5"
    try:
        with open(strModel) as _:
            # Do nothing
            raise FileNotFoundError
            pass
    except FileNotFoundError:
        createkeras_modelFile(strModel, padding="valid")

    from tensorflow import keras

    keras_model = keras.models.load_model(strModel)
    from_model(keras_model, network=torchModel)

    # Initialize keras model from file
    from tensorflow import keras

    keras_model = keras.models.load_model(strModel)
    # Verify no. of layers
    assert len(torchModel.layers) == len(keras_model.layers)


def test_loadkeras_modelSamePadding():
    """
    Test the functionality of loading the model from a keras file
    """
    from sinabs.network import Network
    from sinabs.from_keras import from_model

    # Initialize a Network
    torchModel = Network()
    strModel = strLibPath + "/models/testModelSamePadding.h5"
    # Verify that file exists
    try:
        with open(strModel) as _:
            # Do nothing
            pass
    except FileNotFoundError:
        # Create file if not found
        createkeras_modelFile(strModel, padding="same")

    from tensorflow import keras

    keras_model = keras.models.load_model(strModel)
    from_model(keras_model, network=torchModel)


def test_loadkeras_modelSequential():
    """
    Test the functionality of loading the model from a keras file
    """
    from tensorflow import keras

    inputLayer = keras.layers.InputLayer(input_shape=(20, 24, 3))
    crop = keras.layers.Cropping2D(cropping=((0, 0), (2, 2)))
    conv1 = keras.layers.Conv2D(
        16,
        kernel_size=(4, 5),
        strides=(4, 5),
        padding="same",
        use_bias=False,
        activation="relu",
    )
    conv2 = keras.layers.Conv2D(
        16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        activation="relu",
    )
    pool = keras.layers.AveragePooling2D(pool_size=(2, 2))
    keras_model = keras.Sequential([inputLayer, crop, conv1, conv2, pool])

    from sinabs.network import Network
    from sinabs.from_keras import from_json

    # Initialize a Network
    torchModel = Network()

    from_json(
        keras_json=keras_model.to_json(), input_shape=(3, 20, 24), network=torchModel
    )

    print(torchModel.summary().to_string())


############################################################################################
def createkeras_modelFile(strModel, padding="same"):
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
    output = x
    model = keras.Model(inputs=[inputData], outputs=[output])
    model.save(strModel)
    with open(strModel.strip(".h5") + ".json", "w") as jsonFile:
        jsonFile.write(model.to_json())


def createKerasSeqModelFile(strModel, padding="same"):
    """
    Create a Sequential model file for testing with a diverse set of layers
    """
    from tensorflow import keras

    inputLayer = keras.layers.InputLayer(input_shape=(20, 24, 3))
    crop = keras.layers.Cropping2D(cropping=((0, 0), (2, 2)))
    conv1 = keras.layers.Conv2D(
        16,
        kernel_size=(4, 5),
        strides=(4, 5),
        padding=padding,
        use_bias=False,
        activation="relu",
    )
    conv2 = keras.layers.Conv2D(
        16,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding=padding,
        use_bias=False,
        activation="relu",
    )
    pool = keras.layers.AveragePooling2D(pool_size=(2, 2))
    model = keras.Sequential([inputLayer, crop, conv1, conv2, pool])
    model.save(strModel)
    print(model.summary())
    with open(strModel.strip(".h5") + ".json", "w") as jsonFile:
        jsonFile.write(model.to_json())
