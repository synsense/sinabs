import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.specksim import from_sequential
from sinabs.from_torch import from_model

# Define the path to pre-trained MNIST weights.
weights_path = Path(__file__).absolute().parent / "mnist_params.pt"


# Define custom dataset for spiking input data
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


def convert_raster_to_events(spike_train: torch.tensor):
    dtype = [("t", np.uint32), ("p", np.uint32), ("y", np.uint32), ("x", np.uint32)]
    event_list = []
    max_raster = spike_train.max()
    for val in range(int(max_raster), 0, -1):
        t, ch, y, x = torch.where(spike_train == val)
        ev_data = torch.stack((t, ch, y, x), dim=0).T
        ev_data = ev_data.repeat(val, 1)
        event_list.extend(ev_data)
    ev_data = torch.stack(sorted(event_list, key=lambda event: event[0]), dim=0).numpy()
    return np.array([tuple(x) for x in ev_data], dtype=dtype)


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
ann.load_state_dict(torch.load(weights_path))

## Initialize the dataset
t_window = 100
test_dataset = MNIST_Dataset("./data", train=False, spiking=True, t_window=t_window)
first_sample = convert_raster_to_events(test_dataset[0][0])
first_target = test_dataset[0][1]
print(f"Number of events sent: {len(first_sample)}")

# Sinabs model

## Convert SNN
snn = from_model(ann, add_spiking_output=True, batch_size=1).spiking_model

## Convert to Specksim
specksim_network = from_sequential(snn, input_shape=(1, 28, 28))

# Do the inference
print("Sinabs Network:")
begin_t = time.time()
output_spikes = specksim_network(first_sample)
end_t = time.time()
print(f"Number of events received: {len(output_spikes)}")
print(f"Duration: {end_t - begin_t} seconds.")

# Make a prediction
output_channels = output_spikes["p"]
prediction = np.argmax(np.bincount(output_channels))
print(f"Prediction: {prediction}, target: {first_target}")

# Dynapcnn network

## Convert SNN to Dynapcnn Network
dynapcnn_network = DynapcnnNetwork(
    snn=snn, input_shape=(1, 28, 28), dvs_input=False, discretize=True
)

## Convert Dynapcnn Network to Specksim Network
specksim_network_dynapcnn = from_sequential(dynapcnn_network, input_shape=(1, 28, 28))

# Do the inference
print("Dynapcnn Network - Quantized")
begin_t = time.time()
output_spikes = specksim_network_dynapcnn(first_sample)
end_t = time.time()
print(f"Number of events received: {len(output_spikes)}")
print(f"Duration: {end_t - begin_t} seconds.")

# Make a prediction
output_channels = output_spikes["p"]
prediction = np.argmax(np.bincount(output_channels))
print(f"Prediction: {prediction}, target: {first_target}")
