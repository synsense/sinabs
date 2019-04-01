def test_createModelFromkeras_config_Sequential():
    from tensorflow import keras
    import json

    keras_model = keras.Sequential()
    # Input layer

    keras_model.add(keras.layers.InputLayer(input_shape=(2, 50, 50)))
    keras_model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    keras_model.add(keras.layers.Conv2D(10, kernel_size=(3, 3), padding="same"))

    keras_config = json.loads(keras_model.to_json())
    input_shape = keras_model.get_input_shape_at(0)
    print(input_shape)
    print(keras_config)
    print(keras_model.to_json())
    from sinabs.from_keras import from_model_keras_config

    # Load non spiking model
    myModel = from_model_keras_config(
        keras_config,
        input_shape=input_shape[1:],
        spiking=False,
        quantize_analog_activation=False,
    )
    print(myModel)

    # Load spiking model
    myModelSpiking = from_model_keras_config(
        keras_config,
        input_shape=input_shape[1:],
        spiking=True,
        quantize_analog_activation=False,
    )
    print(myModelSpiking)


def test_createModelFromkeras_config_Model():
    from tensorflow import keras
    import json

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
    keras_config = json.loads(keras_model.to_json())
    input_shape = keras_model.get_input_shape_at(0)

    print(keras_config)

    from sinabs.from_keras import from_model_keras_config

    # Load non spiking model
    myModel = from_model_keras_config(
        keras_config,
        input_shape=input_shape[1:],
        spiking=False,
        quantize_analog_activation=False,
    )
    print(myModel)

    # Load spiking model
    myModelSpiking = from_model_keras_config(
        keras_config,
        input_shape=input_shape[1:],
        spiking=True,
        quantize_analog_activation=False,
    )

    print(myModelSpiking)
    print(list(myModelSpiking.named_children()))


def test_createModelFromkeras_config_channels_last():
    from tensorflow import keras
    import json

    # Input layer
    imgDataFormat = "channels_last"
    inputs = keras.layers.Input(shape=(50, 50, 2))
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

    keras_config = json.loads(keras_model.to_json())
    input_shape = (2, 50, 50)

    print(keras_config)

    from sinabs.from_keras import from_model_keras_config

    # Load non spiking model
    myModel = from_model_keras_config(
        keras_config,
        input_shape=input_shape,
        spiking=False,
        quantize_analog_activation=False,
    )
    print(myModel)

    # Load spiking model
    myModelSpiking = from_model_keras_config(
        keras_config,
        input_shape=input_shape,
        spiking=True,
        quantize_analog_activation=False,
    )

    print(myModelSpiking)
