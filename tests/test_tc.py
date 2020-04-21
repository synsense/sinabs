#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Massimo Bortone).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

def test_import():
    from sinabs.layers import SpikingTemporalConv1dLayer


def test_synaptic_output_shape():
    from sinabs.layers import SpikingTemporalConv1dLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((50, 20)) > 0.05).float()

    # Init layer
    tds = SpikingTemporalConv1dLayer(
        channels_in=20,
        channels_out=5,
        kernel_shape=2,
        dilation=30,
        bias=True
    )
    out_current = tds.synaptic_output(inp_spikes)

    assert out_current.shape == (50, 5)


def test_conv_dims():
    from sinabs.layers import SpikingTemporalConv1dLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((50, 20)) > 0.05).float()

    # Init layer
    tds = SpikingTemporalConv1dLayer(
        channels_in=20,
        channels_out=5,
        kernel_shape=2,
        dilation=30,
        bias=True
    )
    out_spikes = tds(inp_spikes)

    assert out_spikes.shape == (50, 5)




def test_buffer():
    from sinabs.layers import SpikingTemporalConv1dLayer
    import torch
    import numpy as np

    # Generate input
    inp = np.arange(20) % 2
    inp_spikes = torch.from_numpy(inp).float().unsqueeze(dim=1)

    # Init layer
    tds = SpikingTemporalConv1dLayer(
        channels_in=1,
        channels_out=1,
        kernel_shape=2,
        strides=1,
        dilation=2,
        bias=False
    )
    for (key, param) in tds.named_parameters():
        if key == "conv.weight":
            param.data = torch.ones_like(param.data)
        if key == "conv.bias":
            param.data = torch.zeros_like(param.data)
    out_spikes = tds(inp_spikes)
    result = torch.tensor([0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2]).unsqueeze(dim=1)
    assert torch.eq(out_spikes, result).all()

