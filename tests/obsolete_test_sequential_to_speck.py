import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from sinabs import Network
import sinabs.layers as sl
from backend.Speck.tospeck import to_speck_config

plt.ion()


class MySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_1 = sl.InputLayer(input_shape=(1, 28, 28), layer_name="input_1")
        self.conv1 = sl.SpikingConv2dLayer(
            channels_in=1,
            image_shape=(28, 28),
            channels_out=20,
            kernel_shape=(5, 5),
            layer_name="conv1",
            strides=(1, 1),
        )

        self.pool1 = sl.SumPooling2dLayer(
            image_shape=(24, 24), pool_size=(2, 2), layer_name="pool1"
        )
        self.conv2 = sl.SpikingConv2dLayer(
            channels_in=20,
            image_shape=(12, 12),
            channels_out=50,
            kernel_shape=(5, 5),
            layer_name="conv2",
        )

        self.pool2 = sl.SumPooling2dLayer(
            image_shape=(8, 8), pool_size=(2, 2), layer_name="pool2"
        )
        self.fc1 = sl.SpikingConv2dLayer(
            channels_in=50,
            image_shape=(4, 4),
            channels_out=500,
            kernel_shape=(4, 4),
            strides=(4, 4),
            layer_name="fc1",
        )
        self.fc2 = sl.SpikingConv2dLayer(
            channels_in=500,
            image_shape=(1, 1),
            channels_out=10,
            kernel_shape=(1, 1),
            strides=(1, 1),
            layer_name="fc2",
        )

    def forward(self, x):
        x = self.input_1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


snn = Network()
snn.spiking_model = MySNN()
speck_config = to_speck_config(snn)

# Check weight scaling
weights_orig, biases_orig = snn.spiking_model.conv1.parameters()
weights_orig = weights_orig.detach().numpy()
biases_orig = biases_orig.detach().numpy()
threshold_low_orig = snn.spiking_model.conv1.threshold_low
threshold_high_orig = snn.spiking_model.conv1.threshold

weights_speck = np.array(speck_config.cnn_layers[0].get_weights()).transpose(0, 1, 3, 2)
biases_speck = np.array(speck_config.cnn_layers[0].get_biases())
threshold_low_speck = speck_config.cnn_layers[0].threshold_low
threshold_high_speck = speck_config.cnn_layers[0].threshold_high

scale = threshold_low_speck / threshold_low_orig


def error(orig, speck, absolute=False):
    if absolute:
        return np.abs(scale * orig - speck)
    if isinstance(orig, np.ndarray) and isinstance(speck, np.ndarray):
        orig_0 = np.abs(orig * scale) < 0.5
        speck_0 = speck == 0
        if np.any(np.logical_xor(orig_0, speck_0)):
            return 1
        else:
            orig = orig[orig_0 == False]
            speck = speck[speck_0 == False]
    return (scale * orig / speck) - 1


print(f"Higher threshold error: {error(threshold_high_orig, threshold_high_speck):%}")
print(f"Mean weight error: {np.mean(np.abs(error(weights_orig, weights_speck))):%}")
print(f"Mean bias error: {np.mean(np.abs(error(biases_orig, biases_speck))):%}")
print(
    f"Higher threshold absolute error: {error(threshold_high_orig, threshold_high_speck, absolute=True)}"
)
print(
    f"Max weight absolute error: {np.nanmax(np.abs(error(weights_orig, weights_speck, absolute=True)))}"
)
print(
    f"Max bias absolute error: {np.nanmax(np.abs(error(biases_orig, biases_speck, absolute=True)))}"
)

plt.plot(weights_speck.flatten())
plt.plot(weights_orig.flatten() * scale)
