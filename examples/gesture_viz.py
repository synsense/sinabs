import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer

class GestureClassifier(nn.Sequential):
    def __init__(self, num_classes: int):
        super().__init__(
            nn.Conv2d(2, 16, kernel_size=2, stride=2, bias=False),
            nn.ReLU(),
            # core 1
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 7
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 5
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=2, bias=False),
            nn.ReLU(),
            # core 3
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            # core 8
            nn.Linear(128, num_classes, bias=False),
)

num_classes = 17
input_shape = (2, 128, 128)
ann = GestureClassifier(num_classes=num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
ann = ann.to(device)

sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)

load_result = sinabs_model.spiking_model.load_state_dict(torch.load("gesture_model.pth"), strict=False)
print(load_result)

hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    discretize=True,
    input_shape=input_shape,
)

hardware_compatible_model.to(
    device="speck2edevkit:0",
    monitor_layers=["dvs",-1]  # Last layer
)

visualizer = DynapcnnVisualizer(
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    feature_count=17,
    feature_names=[f"{i}" for i in range(17)]
)

visualizer.connect(hardware_compatible_model)

