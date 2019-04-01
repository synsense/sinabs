def test_crop2d():
    from tensorflow import keras

    kerasLayer = keras.layers.Cropping2D(((2, 3), (3, 2)))
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_cropping2d_keras_conf

    # Create spiking layers
    layer_list = from_cropping2d_keras_conf(keras_config, input_shape=(5, 30, 50))

    for layer_name, layer in layer_list:
        print(layer_name)
        print(layer.summary())

    # Verify output shape
    assert layer.output_shape == (5, 25, 45)
