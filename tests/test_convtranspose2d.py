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
    """
    Basic syntax check test
    """
    from sinabs.layers import SpikingConvTranspose2dLayer


def test_TorchSpikingConvTranspose2dLayer_initialization():
    """
    Test initialization of ConvTranspose2dlayer layer
    :return:
    """
    from sinabs.layers import SpikingConvTranspose2dLayer

    lyr = SpikingConvTranspose2dLayer(
        channels_in=2,
        image_shape=(10, 10),
        channels_out=6,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 0, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="convtranspose2d",
    )

    assert lyr.output_shape == (6, 12, 14)


def test_getoutput_shape():
    from sinabs.layers import SpikingConvTranspose2dLayer
    import torch

    lyr = SpikingConvTranspose2dLayer(
        channels_in=2,
        image_shape=(10, 20),
        channels_out=6,
        kernel_shape=(3, 5),
        strides=(1, 1),
        padding=(0, 3, 6, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="convtranspose2d",
    )

    tsrInput = (torch.rand(10, 2, 10, 20) > 0.9).float()

    tsrOutput = lyr(tsrInput)
    assert lyr.output_shape == tsrOutput.shape[1:]
