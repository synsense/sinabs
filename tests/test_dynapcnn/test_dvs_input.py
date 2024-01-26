"""This should test cases of dynapcnn compatible networks with dvs input."""

from itertools import product
from typing import Optional, Tuple

import numpy as np
import pytest
import samna
import torch
from torch import nn

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
from sinabs.backend.dynapcnn.exceptions import *
from sinabs.from_torch import from_model
from sinabs.layers import IAF

INPUT_SHAPE = (2, 16, 16)
input_data = torch.rand(1, *INPUT_SHAPE, requires_grad=False) * 100.0


def combine_n_binary_choices(n):
    return tuple(product(*[(True, False) for __ in range(n)]))


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


class NetNoPooling(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(2, 4, kernel_size=2, stride=2), nn.ReLU()]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class NetPool2D(nn.Module):
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


@pytest.mark.parametrize("dvs_input", (False, True))
def test_dvs_no_pooling(dvs_input):
    # - ANN and SNN generation
    ann = NetNoPooling()
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input, input_shape=INPUT_SHAPE)

    # If there is no pooling, a DVSLayer should only be added if `dvs_input` is True
    assert isinstance(spn.sequence[0], DVSLayer) == dvs_input

    # - Make sure missing input shapes cause exception
    with pytest.raises(InputConfigurationError):
        spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, input_shape=INPUT_SHAPE)
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


@pytest.mark.parametrize("dvs_input", (False, True))
def test_dvs_pooling_2d(dvs_input):
    # - ANN and SNN generation
    ann = NetPool2D(input_layer=True)
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input, input_shape=INPUT_SHAPE)

    # When there is pooling, a DVSLayer should also be added if `dvs_input` is True
    assert isinstance(spn.sequence[0], DVSLayer)

    # - Make sure missing input shapes cause exception
    with pytest.raises(InputConfigurationError):
        spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, input_shape=INPUT_SHAPE)
    snn_out = snn(input_data).squeeze()
    spn_out = spn_float(input_data).squeeze()
    assert torch.equal(snn_out.detach(), spn_out)

    # - Verify DYNAP-CNN config
    target_layers = [5]
    config = spn.make_config(chip_layers_ordering=target_layers)
    verify_dvs_config(
        config,
        input_shape=INPUT_SHAPE,
        destination=target_layers[0],
        dvs_input=dvs_input,
        pooling=(2, 4),
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


@pytest.mark.parametrize("flip_x,flip_y,swap_xy,dvs_input", combine_n_binary_choices(4))
def test_dvs_mirroring(flip_x, flip_y, swap_xy, dvs_input):
    # - DYNAP-CNN layer arrangement
    target_layers = [5]

    kwargs_flip = {"flip_x": flip_x, "flip_y": flip_y, "swap_xy": swap_xy}

    # - ANN and SNN generation
    ann = DvsNet(dvs_input=dvs_input, **kwargs_flip)
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, dvs_input=dvs_input)
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


@pytest.mark.parametrize("dvs_input,merge_polarities", combine_n_binary_choices(2))
def test_dvs_crop(dvs_input, merge_polarities):
    # - DYNAP-CNN layer arrangement
    target_layers = [5]
    crop = ((20, 62), (12, 32))
    pool = (1, 2)

    shape = (1, 128, 128) if merge_polarities else (2, 128, 128)
    input_data = torch.rand(1, *shape, requires_grad=False) * 100.0
    # - ANN and SNN generation
    ann = DvsNet(
        dvs_input=dvs_input,
        crop=crop,
        pool=pool,
        input_shape=shape,
        merge_polarities=merge_polarities,
    )
    snn = from_model(ann.seq, batch_size=1)
    snn.eval()

    # - SPN generation
    spn = DynapcnnNetwork(snn, dvs_input=dvs_input)

    # - Compare snn and spn outputs
    spn_float = DynapcnnNetwork(snn, discretize=False, dvs_input=dvs_input)
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


@pytest.mark.parametrize("dvs_input,pool", combine_n_binary_choices(2))
def test_whether_dvs_mirror_cfg_is_all_switched_off(dvs_input, pool):
    from torch import nn

    from sinabs.backend.dynapcnn import DynapcnnNetwork
    from sinabs.layers import IAFSqueeze, SumPool2d

    layer_list = [SumPool2d(kernel_size=(1, 1))] if pool else []
    layer_list += [
        nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=(16, 16),
            stride=(2, 2),
            padding=(0, 0),
            bias=False,
        ),
        IAFSqueeze(min_v_mem=-1.0, batch_size=1),
    ]

    snn = nn.Sequential(*layer_list)

    dynapcnn = DynapcnnNetwork(
        snn=snn, input_shape=(1, 128, 128), dvs_input=dvs_input, discretize=True
    )
    samna_cfg = dynapcnn.make_config(device="speck2edevkit")

    assert samna_cfg.dvs_layer.pass_sensor_events == dvs_input
    if dvs_input:
        assert samna_cfg.dvs_layer.mirror.x is False
        assert samna_cfg.dvs_layer.mirror.y is False
        assert samna_cfg.dvs_layer.mirror_diagonal is False
