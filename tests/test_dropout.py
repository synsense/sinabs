def test_dropout():
    from tensorflow import keras

    kerasLayer = keras.layers.Dropout(rate=0.5)
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_dropout_keras_conf

    # Create spiking layers
    layer_list = from_dropout_keras_conf(
        keras_config, input_shape=(5, 30, 50), spiking=False
    )

    for layer_name, layer in layer_list:
        print(layer_name)

    # Verify output shape
    assert layer.p == 0.5
