def test_quantize():
    from tensorflow import keras

    kerasLayer = keras.layers.Conv2D(
        3,
        kernel_size=(5, 3),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
    )
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_conv2d_keras_conf

    # Create spiking layers
    layer_list = from_conv2d_keras_conf(
        keras_config,
        input_shape=(5, 30, 50),
        spiking=False,
        quantize_analog_activation=True,
    )

    for layer_name, layer in layer_list:
        print(layer_name)
