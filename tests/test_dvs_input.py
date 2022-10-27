"""
This should test cases of dynapcnn compatible networks with dvs input
"""
import samna

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
from sinabs.backend.dynapcnn.exceptions import *
from sinabs.from_torch import from_model
from sinabs.layers import IAF

import torch
from torch import nn
import numpy as np

from typing import Optional, Tuple
import pytest

INPUT_SHAPE = (2, 16, 16)
input_data = torch.rand(1, *INPUT_SHAPE, requires_grad=False) * 100.0


def verify_dvs_config(
    config,
    input_shape: Tuple[int, int, int],
    pooling: Optional[Tuple[int, int]] = (1, 1),
    origin: Tuple[int, int] = (0, 0),
    cut: Optional[Tuple[int, int]] = None,
    destination: Optional[int] = None,
    dvs_input: bool = True,
    flip: Optional[dict] = None,
    merge_polarities: bool = False,
):
    dvs = config.dvs_layer
    print(dvs.to_json())
    assert dvs.destinations[1].enable is False
    if destination is None:
        assert dvs.destinations[0].enable is False
        return

    if destination is None:
        assert dvs.destinations[0].enable == False
    else:
        assert dvs.destinations[0].enable == True
        assert dvs.destinations[0].layer == destination
    if cut is None:
        assert dvs.cut.y == origin[0] + INPUT_SHAPE[1] // pooling[0] - 1
        assert dvs.cut.x == origin[1] + INPUT_SHAPE[2] // pooling[1] - 1
    else:
        assert dvs.cut.y == cut[0] - 1
        assert dvs.cut.x == cut[1] - 1
    assert dvs.pooling.y == pooling[0]
    assert dvs.pooling.x == pooling[1]
    assert dvs.origin.y == origin[0]
    assert dvs.origin.x == origin[1]
    if flip is None:
        assert not dvs.mirror.x
        assert not dvs.mirror.y
        assert not dvs.mirror_diagonal
    else:
        assert dvs.mirror.x == flip["flip_x"]
        assert dvs.mirror.y == flip["flip_y"]
        assert dvs.mirror_diagonal == flip["swap_xy"]
    assert dvs.pass_sensor_events == dvs_input
    assert dvs.merge == merge_polarities


def test_dvs_no_pooling():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            layers += [nn.Conv2d(2, 4, kernel_size=2, stride=2), nn.ReLU()]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    # - ANN and SNN generation, no input layer
    ann = Net()
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    for dvs_input in (False, True):
        # - SPN generation
        spn = DynapcnnNetwork(
            snn, input_shape=INPUT_SHAPE, dvs_input=dvs_input
        )

        # - Make sure missing input shape causes exception
        with pytest.raises(InputConfigurationError):
            spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

        # - Compare snn and spn outputs
        spn_float = DynapcnnNetwork(
            snn, input_shape=INPUT_SHAPE, dvs_input=dvs_input, discretize=False
        )
        snn_out = snn(input_data).squeeze()
        spn_out = spn_float(input_data).squeeze()
        assert torch.equal(snn_out.detach(), spn_out)

        # - Verify DYNAP-CNN config
        target_layers = [5]
        config = spn.make_config(chip_layers_ordering=target_layers)
        verify_dvs_config(
            config,
            input_shape=INPUT_SHAPE,
            destination=target_layers[0] if dvs_input else None,
            dvs_input=dvs_input,
        )

    # - ANN and SNN generation, network with input layer
    target_layers = [5]
    ann = Net()
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input, input_shape=INPUT_SHAPE)

    ## - Make sure non-matching input shapes cause exception
    #with pytest.raises(InputConfigurationError):
    #    spn = DynapcnnNetwork(snn, dvs_input=dvs_input, input_shape=(1, 2, 3))

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, input_shape=INPUT_SHAPE)
    snn_out = snn(input_data).squeeze()
    spn_out = spn_float(input_data).squeeze()
    assert torch.equal(snn_out.detach(), spn_out)

    # - Verify DYNAP-CNN config
    config = spn.make_config(chip_layers_ordering=target_layers)
    verify_dvs_config(
        config, input_shape=INPUT_SHAPE, destination=target_layers[0], dvs_input=True
    )


def test_dvs_pooling_2d():
    class Net(nn.Module):
        def __init__(self, input_layer: bool = False):
            super().__init__()
            layers = []
            layers += [
                nn.AvgPool2d(kernel_size=(2, 2)),
                nn.AvgPool2d(kernel_size=(1, 2)),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            ]
            self.seq = nn.Sequential(*layers)

        def forward(self, x):
            return self.seq(x)

    pooling = (2, 4)
    target_layers = [5]

    # - ANN and SNN generation, no input layer
    ann = Net()
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    for dvs_input in (False, True):
        # - SPN generation
        spn = DynapcnnNetwork(
            snn, input_shape=INPUT_SHAPE, dvs_input=dvs_input
        )

        # - Make sure missing input shape causes exception
        with pytest.raises(InputConfigurationError):
            spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

        # - Compare snn and spn outputs
        spn_float = DynapcnnNetwork(
            snn, input_shape=INPUT_SHAPE, dvs_input=dvs_input, discretize=False
        )
        snn_out = snn(input_data).squeeze()
        spn_out = spn_float(input_data).squeeze()
        assert torch.equal(snn_out.detach(), spn_out)

        # - Verify DYNAP-CNN config
        config = spn.make_config(chip_layers_ordering=target_layers)
        verify_dvs_config(
            config,
            input_shape=INPUT_SHAPE,
            destination=target_layers[0],
            dvs_input=dvs_input,
            pooling=pooling,
        )

    # - ANN and SNN generation, network with input layer
    ann = Net(input_layer=True)
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input, input_shape=INPUT_SHAPE)

    ## - Make sure non-matching input shapes cause exception
    #with pytest.raises(InputConfigurationError):
    #    spn = DynapcnnNetwork(snn, input_shape=(1, 2, 3))

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, input_shape=INPUT_SHAPE)
    snn_out = snn(input_data).squeeze()
    spn_out = spn_float(input_data).squeeze()
    assert torch.equal(snn_out.detach(), spn_out)

    # - Verify DYNAP-CNN config
    config = spn.make_config(chip_layers_ordering=target_layers)
    verify_dvs_config(
        config,
        input_shape=INPUT_SHAPE,
        destination=target_layers[0],
        dvs_input=True,
        pooling=pooling,
    )


class DvsNet(nn.Module):
    def __init__(
        self,
        pool=(1, 1),
        crop=None,
        dvs_input=False,
        merge_polarities=False,
        input_shape=INPUT_SHAPE,
        **kwargs_flip,
    ):
        super().__init__()
        n_channels_in = 1 if merge_polarities else 2
        layers = [
            DVSLayer(
                input_shape=input_shape[1:],
                pool=pool,
                crop=crop,
                disable_pixel_array=not dvs_input,
                merge_polarities=merge_polarities,
                **kwargs_flip,
            ),
            nn.Conv2d(n_channels_in, 4, kernel_size=2, stride=2),
            IAF(),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


def test_dvs_mirroring():

    # - DYNAP-CNN layer arrangement
    target_layers = [5]

    keys = ["flip_x", "flip_y", "swap_xy"]
    for flag_combination in (
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ):
        kwargs_flip = {key: flag for key, flag in zip(keys, flag_combination)}

        # - ANN and SNN generation
        for dvs_input in (True, False):
            ann = DvsNet(dvs_input=dvs_input, **kwargs_flip)
            snn = from_model(ann.seq, batch_size=1)
            snn.eval()

            # - SPN generation
            spn = DynapcnnNetwork(snn)

            # - Compare snn and spn outputs
            spn_float = DynapcnnNetwork(snn, discretize=False)
            snn_out = snn(input_data).squeeze()
            spn_out = spn_float(input_data).squeeze()
            assert torch.equal(snn_out.detach(), spn_out)

            # - Verify DYNAP-CNN config
            config = spn.make_config(chip_layers_ordering=target_layers)
            verify_dvs_config(
                config,
                input_shape=INPUT_SHAPE,
                destination=target_layers[0],
                dvs_input=dvs_input,
                flip=kwargs_flip,
            )


def test_dvs_crop():

    # - DYNAP-CNN layer arrangement
    target_layers = [5]
    crop = ((20, 62), (12, 32))
    pool = (1, 2)

    for dvs_input in (True, False):
        for merge_polarities in (True, False):
            shape = (1, 128, 128) if merge_polarities else (2, 128, 128)
            input_data = torch.rand(1, *shape, requires_grad=False) * 100.0
            # - ANN and SNN generation
            ann = DvsNet(dvs_input=dvs_input, crop=crop, pool=pool, input_shape=shape, merge_polarities=merge_polarities)
            snn = from_model(ann.seq, batch_size=1)
            snn.eval()

            # - SPN generation
            spn = DynapcnnNetwork(snn)

            # - Compare snn and spn outputs
            spn_float = DynapcnnNetwork(snn, discretize=False)
            snn_out = snn(input_data).squeeze()
            spn_out = spn_float(input_data).squeeze()
            assert torch.equal(snn_out.detach(), spn_out)

            # - Verify DYNAP-CNN config
            config = spn.make_config(chip_layers_ordering=target_layers)
            verify_dvs_config(
                config,
                input_shape=shape,
                pooling=pool,
                origin=(20, 12),
                cut=(62, 32),
                destination=target_layers[0],
                dvs_input=dvs_input,
                merge_polarities=merge_polarities,
            )
