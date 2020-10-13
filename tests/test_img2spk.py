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


def test_img2spk():
    import torch
    from sinabs.layers import Img2SpikeLayer

    lyr = Img2SpikeLayer(
        image_shape=(2, 64, 64), tw=10, max_rate=1000, layer_name="img2spk"
    )

    img = torch.rand(2, 64, 64)

    spks = lyr(img)

    assert spks.shape == (10, 2, 64, 64)
