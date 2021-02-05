"""
This should test cases of dynapcnn compatible networks with dvs input
"""
import samna

from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork
from sinabs.from_torch import from_model
from sinabs.layers import InputLayer

import torch
from torch import nn
import numpy as np

from typing import Optional, Tuple
import pytest

INPUT_SHAPE = (2, 16, 16)
input_data = torch.rand(1, *INPUT_SHAPE, requires_grad=False) * 100.0


def verify_dvs_config(
    config,
    pooling: Optional[Tuple[int]] = (1, 1),
    input_shape: Optional[Tuple[int]] = False,
    destination: Optional[int] = None,
):
    dvs = config.dvs_layer
    assert dvs.destinations[1].enable is False
    if destination is None:
        assert dvs.destinations[0].enable is False
        return

    assert dvs.destinations[0].enable is True
    assert dvs.destinations[0].layer == destination
    assert dvs.cut.y == INPUT_SHAPE[1] - 1
    assert dvs.cut.x == INPUT_SHAPE[2] - 1
    assert dvs.pooling.y == pooling[0]
    assert dvs.pooling.x == pooling[1]


def verify_networks(
    ann, target_layers, pooling, discretize, input_shape=INPUT_SHAPE, first_pooling=True
):

    # - ANN and SNN generation
    snn = from_model(ann)

    snn.eval()
    snn_out = snn(input_data).squeeze()

    # - SPN generation and output comparison
    snn.reset_states()
    spn = DynapcnnCompatibleNetwork(
        snn, input_shape=input_shape, discretize=discretize, dvs_input=True
    )

    spn_out = spn(input_data).squeeze()
    if not discretize:
        assert np.array_equal(snn_out.detach(), spn_out)

    # - Version without dvs
    if first_pooling:
        with pytest.raises(TypeError):
            DynapcnnCompatibleNetwork(
                snn, input_shape=input_shape, discretize=discretize, dvs_input=False
            )
    else:
        spn_no_dvs = DynapcnnCompatibleNetwork(
            snn, input_shape=input_shape, discretize=discretize, dvs_input=False
        )
        spn_out_no_dvs = spn_no_dvs(input_data).squeeze()
        if not discretize:
            assert np.array_equal(snn_out.detach(), spn_out_no_dvs)

    # - DYNAP-CNN config
    config = spn.make_config(target_layers)
    verify_dvs_config(config, pooling=pooling, input_shape=input_shape, destination=target_layers[0])
    if not first_pooling:
        config_no_dvs = spn_no_dvs.make_config(target_layers)
        verify_dvs_config(config_no_dvs, input_shape=input_shape, destination=None)



def test_dvs_no_pooling():
    class Net(nn.Module):
        def __init__(self, input_layer: bool = False):
            super().__init__()
            layers = [InputLayer(INPUT_SHAPE)] if input_layer else []
            layers += [nn.Conv2d(2, 4, kernel_size=2, stride=2), nn.ReLU()]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    # - DYNAP-CNN layer arrangement
    target_layers = [5, 2]
    pooling = (1, 1)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False, first_pooling=False)
    verify_networks(net, target_layers, pooling, discretize=True, first_pooling=False)
    # - Make sure missing input shape causes exception
    with pytest.raises(ValueError):
        verify_networks(
            net,
            target_layers,
            pooling,
            discretize=False,
            first_pooling=False,
            input_shape=None,
        )
        verify_networks(
            net,
            target_layers,
            pooling,
            discretize=True,
            first_pooling=False,
            input_shape=None,
        )

    # - Network starting with input layer
    net_input_layer = Net(input_layer=True)
    verify_networks(
        net_input_layer,
        target_layers,
        pooling,
        discretize=False,
        first_pooling=False,
        input_shape=None,
    )
    verify_networks(
        net_input_layer,
        target_layers,
        pooling,
        discretize=True,
        first_pooling=False,
        input_shape=None,
    )
    # - Make sure non-matching input shapes cause warning
    with pytest.warns(UserWarning):
        verify_networks(
            net_input_layer,
            target_layers=target_layers,
            pooling=pooling,
            discretize=False,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )
        verify_networks(
            net_input_layer,
            target_layers=target_layers,
            pooling=pooling,
            discretize=True,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )


def test_dvs_pooling_2d():
    class Net(nn.Module):
        def __init__(self, input_layer: bool = False):
            super().__init__()
            layers = [InputLayer(INPUT_SHAPE)] if input_layer else []
            layers += [
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.AvgPool2d(kernel_size=(1, 2)),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            ]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    # - DYNAP-CNN layer arrangement
    target_layers = [5, 2]
    pooling = (2, 4)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False)
    verify_networks(net, target_layers, pooling, discretize=True)
    # - Make sure missing input shape causes exception
    with pytest.raises(ValueError):
        verify_networks(
            net,
            target_layers,
            pooling,
            discretize=False,
            first_pooling=False,
            input_shape=None,
        )
        verify_networks(
            net,
            target_layers,
            pooling,
            discretize=True,
            first_pooling=False,
            input_shape=None,
        )

    net_input_layer = Net(input_layer=True)
    verify_networks(
        net_input_layer, target_layers, pooling, discretize=False, input_shape=None
    )
    verify_networks(
        net_input_layer, target_layers, pooling, discretize=True, input_shape=None
    )
    # - Make sure non-matching input shapes cause warning
    with pytest.warns(UserWarning):
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=False,
            first_pooling=True,
            input_shape=(1, 2, 3),
        )
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=True,
            first_pooling=True,
            input_shape=(1, 2, 3),
        )


def test_dvs_pooling_1d():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    # - DYNAP-CNN layer arrangement
    target_layers = [5, 2]
    pooling = (4, 4)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False)
    verify_networks(net, target_layers, pooling, discretize=True)


def test_dvs_pooling_1d2d():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    # - DYNAP-CNN layer arrangement
    target_layers = [5, 2]
    pooling = (4, 2)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False)
    verify_networks(net, target_layers, pooling, discretize=True)


def test_dvs_pooling_2d1d():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, 1)),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    # - DYNAP-CNN layer arrangement
    target_layers = [5, 2]
    pooling = (4, 2)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False)
    verify_networks(net, target_layers, pooling, discretize=True)
