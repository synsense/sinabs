import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets

from sinabs.backend.dynapcnn.specksim import from_sinabs
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.from_torch import from_model

# Define custom dataset for spiking input data
class MNIST_Dataset(datasets.MNIST):

    def __init__(self, root, train=True, spiking=False, t_window=100):
        super().__init__(root, train=train, download=True)
        self.spiking = spiking
        self.t_window = t_window

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.spiking:
            img = (np.random.rand(self.t_window, 1, *img.size()) < img.numpy() / 255.0).astype(float)
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
ann.load_state_dict(torch.load("./tests/mnist_params.pt"))

# Convert SNN
snn = from_model(ann, add_spiking_output=True, batch_size=1).spiking_model

# Convert to Specksim
specksim_network = from_sinabs(snn, input_shape=(1, 28, 28))
specksim_network.sleep_duration = 1.0

# Initialize the dataset
t_window = 200
test_dataset = MNIST_Dataset("./data", train=False, spiking=True, t_window=t_window)
first_sample = convert_raster_to_events(test_dataset[0][0])
first_target = test_dataset[0][1]


# Do the inference
output_spikes = specksim_network(first_sample)
print(output_spikes)