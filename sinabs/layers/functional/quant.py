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

import torch.autograd


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        """"""
        return inp.floor()

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_input = grad_output.clone()
        return grad_input


class StochasticRounding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        """"""
        int_val = inp.floor()
        frac = inp - int_val
        output = int_val + (torch.rand_like(inp) < frac).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_input = grad_output.clone()
        return grad_input


quantize = Quantize().apply
stochastic_rounding = StochasticRounding().apply
