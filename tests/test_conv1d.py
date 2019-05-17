#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
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
    from sinabs.layers import SpikingConv1dLayer


def test_init():
    from sinabs.layers import SpikingConv1dLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((100, 5, 100)) > 0.95).float()

    # Init layer
    conv = SpikingConv1dLayer(
        channels_in=5,
        image_shape=100,
        channels_out=7,
        kernel_shape=5,
        padding=(0, 1),
    )

    out_spikes = conv(inp_spikes)

    assert out_spikes.shape == (100, 7, 97)
