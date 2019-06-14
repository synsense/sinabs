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
    from sinabs.layers import SpikingTDSLayer


def test_synaptic_output():
    from sinabs.layers import SpikingTDSLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((50, 20, 1)) > 0.05).float()

    # Init layer
    tds = SpikingTDSLayer(
        channels_in=20,
        channels_out=5,
        delay=30,
        bias=True
    )
    out_current = tds.synaptic_output(inp_spikes)

    assert out_current.shape == (50, 5, 1)


def test_conv():
    from sinabs.layers import SpikingTDSLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((50, 20, 1)) > 0.05).float()

    # Init layer
    tds = SpikingTDSLayer(
        channels_in=20,
        channels_out=5,
        delay=30,
        bias=True
    )
    out_spikes = tds(inp_spikes)

    assert out_spikes.shape == (50, 5, 1)


def test_gpu():
    from sinabs.layers import SpikingTDSLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((50, 20, 1)) > 0.05).float().to("cuda:0")

    # Init layer
    tds = SpikingTDSLayer(
        channels_in=20,
        channels_out=5,
        delay=30,
        bias=True
    )
    tds.to("cuda:0")
    out_spikes = tds(inp_spikes)

    assert out_spikes.shape == (50, 5, 1)