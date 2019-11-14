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
import torch.nn as nn


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


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
            return _Quantize.apply(data)
        else:
            return data


class NeuromorphicReLU(torch.nn.Module):
    """
    NeuromorphicReLU layer. This layer is NOT used for Sinabs networks; it's
    useful while training analogue pyTorch networks for future use with Sinabs.

    :param quantize: Whether or not to quantize the output (i.e. floor it to \
    the integer below), in order to mimic spiking behavior.
    :param fanout: Useful when computing the number of SynOps of a quantized \
    NeuromorphicReLU. The activity can be accessed through \
    NeuromorphicReLU.activity, and is multiplied by the value of fanout.
    """
    def __init__(self, quantize=True, fanout=1):
        super().__init__()
        self.quantize = quantize
        self.fanout = fanout

    def forward(self, input):
        output = torch.nn.functional.relu(input)

        if self.quantize:
            output = _Quantize.apply(output)

        self.activity = output.sum() / len(output) * self.fanout
        return output
