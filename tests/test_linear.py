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


def test_spiking_linear_layer_init():
    import sinabs.layers as sl
    import torch

    lyr = sl.SpikingLinearLayer(in_features=10, out_features=20, bias=True)
    # With bias true, there is constant injection current.
    # Define input of zeros
    inp = torch.zeros(100, 10)
    # Output
    out = lyr(inp)

    print(out.shape, torch.nonzero(out))

    assert out.shape[1] == 20

