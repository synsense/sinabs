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

import torch


class SumPool2d(torch.nn.LPPool2d):
    """
    Non-spiking sumpooling layer to be used in analogue Torch models. It is identical to torch.nn.LPPool2d with p=1.

    :param kernel_size: the size of the window
    :param stride: the stride of the window. Default value is kernel_size
    :param ceil_mode: when True, will use ceil instead of floor to compute the output shape
    """
    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super().__init__(1, kernel_size, stride, ceil_mode)
