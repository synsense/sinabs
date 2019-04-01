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
    spkModel = Network(keras_model=keras_model)


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
    spkModel = Network(keras_model=keras_model)

    print(spkModel.summary().to_string())
