import os
import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer

ann = nn.Sequential(
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
    nn.Linear(128, 11, bias=False),
)
load_result = ann.load_state_dict(torch.load("dvs_gesture_params.pt"), strict=False)
print(load_result)
with torch.no_grad():
    ann[0].weight *= 4 # scale first layer weights to get more activity
sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)

input_shape = (2, 128, 128)
hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    discretize=True,
    input_shape=input_shape,
)

hardware_compatible_model.to(
    device="speck2b",
    monitor_layers=["dvs", -1],  # Last layer
    chip_layers_ordering="auto"
)

icons_folder_path = str(os.path.abspath(__file__)).split("/")[:-1]
icons_folder_path = os.path.join("/", os.path.join(*icons_folder_path),"icons")

visualizer = DynapcnnVisualizer(
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    add_readout_plot=True,
    spike_collection_interval=2000,
    feature_count=11,
    readout_images=sorted([os.path.join(icons_folder_path, f) for f in os.listdir(icons_folder_path)])
)
visualizer.connect(hardware_compatible_model)

import samna
import time
graph = samna.graph.EventFilterGraph()
(_, evf, buff) = graph.sequential(
    [
        hardware_compatible_model.samna_device.get_model_source_node(),
        samna.graph.JitMemberSelect(),
        samna.BasicSinkNode_speck2e_event_output_event()
    ]
)
evf.set_white_list([0, 1, 2, 3, 4, 5, 6, 7, 8], "layer")
while True:
    evs = buff.get_events()
    print(len(evs))
    time.sleep(0.5)
