import pytest
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.mapping import make_flow_graph, edmonds, recover_mapping

ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)
input_shape = (1, 28, 28)


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_multi_device_movement():
    hardware_compatible_model = DynapcnnNetwork(
        sinabs_model.spiking_model.cpu(),
        discretize=True,
        input_shape=input_shape,
    )

    hardware_compatible_model.to("speck2b:0")

    print("Second attempt")
    hardware_compatible_model.to("speck2b:0")