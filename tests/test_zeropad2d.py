def test_sumpool2d():
    from tensorflow import keras

    kerasLayer = keras.layers.ZeroPadding2D(padding=((4, 3), (2, 3)))
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_zeropad2d_keras_conf

    # Create spiking layers
    layer_list = from_zeropad2d_keras_conf(
        keras_config, input_shape=(5, 30, 50), spiking=True
    )

    for layer_name, layer in layer_list:
        print(layer_name)
        print(layer.summary())

    # Verify output shape
    assert layer.output_shape == (5, 37, 55)
