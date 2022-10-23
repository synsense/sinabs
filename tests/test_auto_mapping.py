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

hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    discretize=True,
    input_shape=input_shape,
)


def test_auto_mapping():
    config = hardware_compatible_model.make_config("auto", device="dynapcnndevkit")


def test_auto_mapping_should_not_work():
    layer_mapping = [[1, 2], [1, 2], [1, 2]]

    graph = make_flow_graph(layer_mapping)
    new_graph = edmonds(graph, 0, len(graph)-1)
    with pytest.raises(ValueError):
        mapping = recover_mapping(new_graph, layer_mapping)
