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


def test_quantize():
    import torch
    from sinabs.layers import NeuromorphicReLU

    x = torch.rand(20, requires_grad=True)
    lyr = NeuromorphicReLU(quantize=True, fanout=1, stochastic_rounding=False)

    interm = lyr(x)
    out = 40*interm
    err = (out.sum()-100)**2
    err.backward()
    assert out.sum() == 0
    assert x.grad.sum() != 0


def test_stochastic_rounding():
    import torch
    from sinabs.layers import NeuromorphicReLU

    x = torch.rand(20, requires_grad=True)
    lyr = NeuromorphicReLU(quantize=True, fanout=1, stochastic_rounding=True)

    interm = lyr(x)
    out = 40*interm
    err = (out.sum()-100)**2
    err.backward()
    assert out.sum() > 0
    assert x.grad.sum() != 0
