"""
This should test cases of speck compatible networks with dvs input
"""
try:
    from samna.speck.configuration import SpeckConfiguration
except (ImportError, ModuleNotFoundError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True

from sinabs.backend.speck import SpeckCompatibleNetwork
from sinabs.from_torch import from_model
from sinabs.layers import InputLayer

import torch
from torch import nn
import numpy as np

from typing import Optional, Tuple
from warnings import warn
import pytest

input_shape = (2, 16, 16)
input_data = torch.rand(1, *input_shape, requires_grad=False) * 100.0


def verify_dvs_config(
    config: SpeckConfiguration,
    pooling: Optional[Tuple[int]] = (1, 1),
    input_shape: Optional[Tuple[int]] = None,
    destination: Optional[int] = None,
):
    dvs = config.dvs_layer
    assert dvs.destinations[1].enable == False
    if destination is None:
        assert dvs.destinations[0].enable == False
        return

    assert dvs.destinations[0].enable == True
    assert dvs.destinations[0].layer == destination
    assert dvs.cut.y == input_shape[1]
    assert dvs.cut.x == input_shape[2]
    assert dvs.pooling.y == pooling[0]
    assert dvs.pooling.x == pooling[1]


def verify_networks(
    ann, target_layers, pooling, discretize, input_shape=input_shape, first_pooling=True
):

    # - ANN and SNN generation
    snn = from_model(ann)

    snn.eval()
    snn_out = snn(input_data).squeeze()

    # - SPN generation and output comparison
    snn.reset_states()
    spn = SpeckCompatibleNetwork(
        snn, input_shape=input_shape, discretize=discretize, dvs_input=True
    )

    spn_out = spn(input_data).squeeze()
    if not discretize:
        assert np.array_equal(snn_out, spn_out)

    # - Version without dvs
    if first_pooling:
        with pytest.raises(TypeError):
            SpeckCompatibleNetwork(
                snn, input_shape=input_shape, discretize=discretize, dvs_input=False
            )
    else:
        spn_no_dvs = SpeckCompatibleNetwork(
            snn, input_shape=input_shape, discretize=discretize, dvs_input=False
        )
        spn_out_no_dvs = spn_no_dvs(input_data).squeeze()
        if not discretize:
            assert np.array_equal(snn_out, spn_out_no_dvs)

    # - Speck config
    if SAMNA_AVAILABLE:
        config = spn.make_config(target_layers)
        verify_dvs_config(config, pooling, input_shape, target_layers[0])
        if not first_pooling:
            config_no_dvs = spn_no_dvs.make_config(target_layers)
            verify_dvs_config(config_no_dvs, destination=None)
    else:
        warn("Samna not available. Could not perform all tests.")


def test_dvs_no_pooling():
    class Net(nn.Module):
        def __init__(self, input_layer: bool = False):
            super().__init__()
            layers = [InputLayer(input_shape)] if input_layer else []
            layers += [nn.Conv2d(2, 4, kernel_size=2, stride=2), nn.ReLU()]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    # - Speck layer arrangement
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
    with pytest.raises(Warning):
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=False,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=True,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )


def test_dvs_pooling_2d():
    class Net(nn.Module):
        def __init__(self, input_layer: bool = False):
            super().__init__()
            layers = [InputLayer(input_shape)] if input_layer else []
            layers += [
                nn.AvgPool2d(kernel_size=(2, 4)),
                nn.AvgPool2d(kernel_size=(1, 2)),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            ]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    # - Speck layer arrangement
    target_layers = [5, 2]
    pooling = (2, 8)

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
    with pytest.raises(Warning):
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=False,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )
        verify_networks(
            net_input_layer,
            target_layers,
            pooling,
            discretize=True,
            first_pooling=False,
            input_shape=(1, 2, 3),
        )


def test_dvs_pooling_1d():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.AvgPool2d(kernel_size=4),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    # - Speck layer arrangement
    target_layers = [5, 2]
    pooling = (8, 8)

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

    # - Speck layer arrangement
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

    # - Speck layer arrangement
    target_layers = [5, 2]
    pooling = (4, 2)

    net = Net()
    verify_networks(net, target_layers, pooling, discretize=False)
    verify_networks(net, target_layers, pooling, discretize=True)
