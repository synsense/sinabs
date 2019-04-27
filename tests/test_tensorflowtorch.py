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
import numpy as np

#
strLibPath = sys.path[0]
sys.path.insert(1, strLibPath)


INPUT_SIZE = (16, 16)
modelName = "testTorchKerasActivations"


def test_KerasTorchAnalog_channels_last():
    from tensorflow import keras
    import torch

    from sinabs.network import Network
    from sinabs.from_keras import from_json, infer_data_format

    img_data_format = "channels_last"

    # Create a test model
    createTestModel(img_data_format=img_data_format)

    keras_model = keras.models.load_model(f"{strLibPath}/models/{modelName}.h5")
    data_format = infer_data_format(keras_model.get_config())
    # Initialize a sinabs model
    dynapModel = Network()
    with open(f"{strLibPath}/models/{modelName}.json") as jsonfile:
        for json in jsonfile.readlines():
            keras_json = json
    from_json(keras_json=keras_json, network=dynapModel)
    weights = []
    data = np.load(f"{strLibPath}/weights/{modelName}.npz")
    for w in data.files:
        weights.append(data[w])
    dynapModel.set_weights(
        weights=weights, is_from_keras=True, img_data_format=data_format
    )

    # Create input
    tsrInp: torch.Tensor = torch.rand((1, 1, *INPUT_SIZE)).float()
    arrInp = tsrInp.numpy().copy()

    if img_data_format == "channels_last":
        arrInp = arrInp.transpose((0, 2, 3, 1))

    # Process data through torch model
    tsrOut = dynapModel.analog_model(tsrInp)

    # Process data through keras model
    arrOut = keras_model.predict(arrInp)

    assert tsrOut.shape == arrOut.shape

    fMeanErr = np.abs(np.mean(tsrOut.detach().numpy() - arrOut))
    fMeanVal = 0.5 * np.mean(np.abs(tsrOut.detach().numpy()) + np.abs(arrOut))
    # assert fMeanErr == 0
    print(tsrOut, arrOut)
    print(fMeanErr, fMeanVal)
    assert fMeanErr / fMeanVal < 1e-5


def test_KerasTorchAnalog_channels_first():
    from tensorflow import keras
    import torch

    from sinabs.network import Network
    from sinabs.from_keras import from_json, infer_data_format

    img_data_format = "channels_first"

    # Create a test model
    createTestModel(img_data_format=img_data_format)

    keras_model = keras.models.load_model(f"{strLibPath}/models/{modelName}.h5")
    data_format = infer_data_format(keras_model.get_config())
    # Initialize a sinabs model
    dynapModel = Network()
    with open(f"{strLibPath}/models/{modelName}.json") as jsonfile:
        for json in jsonfile.readlines():
            keras_json = json
    from_json(keras_json=keras_json, network=dynapModel)
    weights = []
    data = np.load(f"{strLibPath}/weights/{modelName}.npz")
    for w in data.files:
        weights.append(data[w])
    dynapModel.set_weights(
        weights=weights, is_from_keras=True, img_data_format=data_format
    )

    # Create input
    tsrInp: torch.Tensor = torch.rand((1, 1, *INPUT_SIZE)).float()
    arrInp = tsrInp.numpy().copy()

    if img_data_format == "channels_last":
        arrInp = arrInp.transpose((0, 2, 3, 1))

    # Process data through torch model
    tsrOut = dynapModel.analog_model(tsrInp)

    # Process data through keras model
    arrOut = keras_model.predict(arrInp)

    assert tsrOut.shape == arrOut.shape

    fMeanErr = np.abs(np.mean(tsrOut.detach().numpy() - arrOut))
    fMeanVal = 0.5 * np.mean(np.abs(tsrOut.detach().numpy()) + np.abs(arrOut))
    # assert fMeanErr == 0
    print(tsrOut, arrOut)
    print(fMeanErr, fMeanVal)
    assert fMeanErr / fMeanVal < 1e-5


def createTestModel(img_data_format="channels_last", use_bias=False):
    from tensorflow import keras

    if img_data_format == "channels_last":
        inputLayer = keras.Input((*INPUT_SIZE, 1))
    else:
        inputLayer = keras.Input((1, *INPUT_SIZE))

    x = keras.layers.Conv2D(
        6,
        kernel_size=(4, 4),
        # strides=(2, 2),
        padding="valid",
        data_format=img_data_format,
        activation="relu",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.RandomNormal(mean=-0.0, stddev=0.05),
    )(inputLayer)
    x = keras.layers.Conv2D(
        16,
        kernel_size=(4, 4),
        # strides=(2, 2),
        padding="valid",
        data_format=img_data_format,
        activation="relu",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.RandomNormal(mean=-0.0, stddev=0.05),
    )(x)
    x = keras.layers.Conv2D(
        3,
        kernel_size=(4, 4),
        # strides=(2, 2),
        padding="valid",
        data_format=img_data_format,
        activation="relu",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.RandomNormal(mean=-0.0, stddev=0.05),
    )(x)
    x = keras.layers.Conv2D(
        20,
        kernel_size=(4, 4),
        # strides=(2, 2),
        padding="valid",
        data_format=img_data_format,
        activation="relu",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.RandomNormal(mean=-0.0, stddev=0.05),
    )(x)
    x = keras.layers.Flatten(data_format=img_data_format)(x)
    # x = keras.layers.Dense(
    #    5,
    #    activation="relu",
    #    use_bias=use_bias,
    #    kernel_initializer=keras.initializers.RandomNormal(mean=-0.0, stddev=0.05),
    # )(x)

    output = x
    keras_model = keras.Model(inputLayer, output)
    weights = keras_model.get_weights()
    myweights = []
    for w in weights:
        print("Weights :", w.shape, np.mean(w), np.std(w))
    keras_model.set_weights(myweights)
    print(keras_model.summary())
    keras_model.save(f"{strLibPath}/models/{modelName}.h5")

    from sinabs.from_keras import extract_json_weights

    # Extract json and weight files from the model
    extract_json_weights(
        strModelFile=f"{strLibPath}/models/{modelName}.h5", strPath=strLibPath
    )
