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

import torch.nn as nn
from .functional import quantize


class QuantizeLayer(nn.Module):
    """
    Layer that quantizes the input, i.e. returns floor(input).

    :param quantize: If False, this layer will do nothing.
    """

    def __init__(self, quantize=True):
        super().__init__()
        self.quantize = quantize

    def forward(self, data):
        if self.quantize:
            return quantize(data)
        else:
            return data
