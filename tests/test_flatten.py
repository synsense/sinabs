def test_flatten():
    from tensorflow import keras

    kerasLayer = keras.layers.Flatten()
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_flatten_keras_conf

    # Create spiking layers
    layer_list = from_flatten_keras_conf(
        keras_config, input_shape=(5, 30, 50), spiking=False
    )

    for layer_name, layer in layer_list:
        print(layer_name)
        print(layer.summary())

    # Verify output shape
    assert layer.output_shape == (5 * 30 * 50,)
