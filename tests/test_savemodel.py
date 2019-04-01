import sys

strLibPath = sys.path[0] + "/../"


def test_Generatesinabs_ModelFile():
    """
    Test the functionality of loading the model from a keras file
    """
    # Create a keras model
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
    output = x
    keras_model = keras.Model(inputs=[inputData], outputs=[output])

    # Load model into library
    from sinabs.network import Network
    from sinabs.from_keras import from_model

    # Initialize a Network
    dynapcnn = Network()

    from_model(keras_model=keras_model, input_shape=(3, 20, 24), network=dynapcnn)
    print("--------------------")
    print(dynapcnn.summary().to_string())

    from sinabs.savemodel import to_json

    print(to_json(dynapcnn))
