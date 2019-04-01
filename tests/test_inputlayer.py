def test_inputlayer():
    from tensorflow import keras

    inpLayer = keras.layers.InputLayer(input_shape=(5, 20, 20))
    kerasConf = inpLayer.get_config()

    from sinabs.layers import get_input_shape_from_keras_conf

    input_shape = get_input_shape_from_keras_conf(kerasConf)
    assert input_shape == (5, 20, 20)
