import time
from pathlib import Path

import numpy as np
import samna
import torch
import torch.nn as nn
from torchvision import datasets

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.io import close_device
from sinabs.from_torch import from_model

# Define the path to pre-trained MNIST weights.
weights_path = Path(__file__).absolute().parent / "mnist_params.pt"

# Define custom dataset for spiking input data
input_shape = (1, 28, 28)


class MNIST_Dataset(datasets.MNIST):
    def __init__(self, root, train=True, spiking=False, t_window=100):
        super().__init__(root, train=train, download=True)
        self.spiking = spiking
        self.t_window = t_window

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.spiking:
            img = (
                np.random.rand(self.t_window, 1, *img.size()) < img.numpy() / 255.0
            ).astype(float)
            img = torch.from_numpy(img).float()
        else:
            # Convert image to tensor
            img = torch.from_numpy(img.numpy()).float()
            img.unsqueeze_(0)

        return img, target


# Define ANN
ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

device = "cpu"
ann.to(device)
ann.load_state_dict(torch.load(weights_path, map_location=device))

# Scale weights
with torch.no_grad():
    ann[0].weight.data *= 0.5

# Convert SNN
sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1).spiking_model

# Convert SNN to
hardware_compatible_model = DynapcnnNetwork(
    sinabs_model,
    discretize=True,
    input_shape=input_shape,
)

# Chip name
# Depending on the available hardware you have, you can choose to run the
# network for example on 'dynapcnndevkit' or 'speck2b' among others. See
# documentation for full list of supported devices
# chip_name = "speck2fmodule"
chip_name = "dynapcnndevkit"

hardware_compatible_model.to(
    device=chip_name, monitor_layers=[-1]  # Monitor the output layer
)

# Define dataloader
t_window = 100  # ms (or) time steps

# Initialize the dataset and get the first sample and target
test_dataset = MNIST_Dataset("./data", train=False, spiking=True, t_window=t_window)
first_sample = test_dataset[0][0]
first_target = test_dataset[0][1]
# Check the chip layer ordering to send the events to the correct core
chip_layers_ordering = hardware_compatible_model.chip_layers_ordering
print(
    f"The model was placed on the chip at the following cores: {chip_layers_ordering}"
)
# Convert the spike train raster to chip events
factory = ChipFactory(chip_name)
input_events = factory.raster_to_events(
    first_sample, layer=chip_layers_ordering[0], dt=0.001  # First layer on the chip
)

# Process events
print("Sending events to device")
evs_out = hardware_compatible_model(input_events)

# Filter readout events
if chip_name == "dynapcnndevkit":
    evs_out = list(filter(lambda x: isinstance(x, samna.dynapcnn.event.Spike), evs_out))

if len(evs_out) > 0:
    print(
        f"{len(evs_out)} output events read from time {evs_out[0].timestamp} to {evs_out[-1].timestamp}"
    )

    # Reading events out of the device
    for ev in evs_out:
        print(ev.feature, ev.timestamp)
else:
    print("No output events received!")

close_device(chip_name)
