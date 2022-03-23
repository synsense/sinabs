import time
import torch
import samna
import numpy as np
import torch.nn as nn
from torchvision import datasets
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import io
from sinabs.backend.dynapcnn import DynapcnnNetwork

from sinabs.backend.dynapcnn.chip_factory import ChipFactory

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

device = "cuda" if torch.cuda.is_available() else "cpu"
ann = ann.to(device)

ann.load_state_dict(torch.load("mnist_params.pt", map_location=device))

sinabs_model = from_model(ann, add_spiking_output=True)

input_shape = (1, 28, 28)

hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    discretize=True,
    input_shape=input_shape,
)

# Depending on the available hardware you have, you can choose to run the
# network for example on 'dynapcnndevkit' or 'speck2b' among others. See
# documentation for full list of supported devices
hardware_compatible_model.to(
    device="dynapcnndevkit:0",
    monitor_layers=[-1]  # Last layer
)


# Define custom dataset for spiking input data
class MNIST_Dataset(datasets.MNIST):

    def __init__(self, root, train=True, spiking=False, tWindow=100):
        super().__init__(root, train=train, download=True)
        self.spiking = spiking
        self.tWindow = tWindow

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.spiking:
            img = (np.random.rand(self.tWindow, 1, *img.size()) < img.numpy() / 255.0).astype(float)
            img = torch.from_numpy(img).float()
        else:
            # Convert image to tensor
            img = torch.from_numpy(img.numpy()).float()
            img.unsqueeze_(0)

        return img, target


# Define dataloader
tWindow = 200  # ms (or) time steps

# Define test dataset loader
test_dataset = MNIST_Dataset("./data", train=False, spiking=True, tWindow=tWindow)
chip_layers_ordering = hardware_compatible_model.chip_layers_ordering
print(f"The model was placed on the chip at the following layers: {chip_layers_ordering}")


factory = ChipFactory("dynapcnndevkit:0")
# Generate input events
input_events = factory.raster_to_events(
    test_dataset[0][0],
    layer=chip_layers_ordering[0],  # First layer on the chip
)

# Flush buffer
hardware_compatible_model.samna_output_buffer.get_events()

# Process events
print("Sending events to device")
evs_out = hardware_compatible_model(input_events)
print(f"{len(evs_out)} output events read from time {evs_out[0].timestamp} to {evs_out[-1].timestamp}")

# Filter readout events
evs_out = list(filter(lambda x: isinstance(x, samna.dynapcnn.event.Spike), evs_out))

# Reading events out of the device
for ev in evs_out:
    print(ev.feature, ev.timestamp)

io.close_device("dynapcnndevkit")
