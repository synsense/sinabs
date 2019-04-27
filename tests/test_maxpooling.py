def test_maxpool2d():
    from tensorflow import keras

    kerasLayer = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=None)
    keras_config = kerasLayer.get_config()

    from sinabs.layers import from_maxpool2d_keras_conf

    # Create spiking layers
    layer_list = from_maxpool2d_keras_conf(
        keras_config, input_shape=(5, 30, 50), spiking=True
    )

    for layer_name, layer in layer_list:
        print(layer_name)
        print(layer.summary())

    # Verify output shape
    assert layer.output_shape == (5, 10, 16)


def test_maxpool_function():
    from sinabs.layers import SpikingMaxPooling2dLayer
    import torch

    lyr = SpikingMaxPooling2dLayer(image_shape=(2, 3), pool_size=(2, 3), strides=None)

    tsrInput = (torch.rand(10, 1, 2, 3) > 0.8).float()
    # print(tsrInput.sum(0))
    tsrInput[:, 0, 0, 2] = 1
    tsrOut = lyr(tsrInput)

    assert tsrOut.sum().item() == 10.0

    from tensorflow import keras

    kerasLayer = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None)
    keras_config = kerasLayer.get_config()

    # Create spiking layers
