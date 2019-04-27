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

from typing import Tuple


def conv_output_size(image_length: int, kernel_length: int, stride: int) -> int:
    """
    Computes output dimension given input dimension, kernel size and stride,
    assumign no padding, *per* dimension given

    :param image_length: int image size on one dimension
    :param kernel_length: int kernel_length size on one dimension
    :param stride: int Stride size on one dimension
    :return: int -- convolved output image size on one dimension

    """
    try:
        assert image_length >= kernel_length
    except AssertionError:
        raise Exception(
            "Image dimension {0} smaller than kernel dimension {1}".format(
                image_length, kernel_length
            )
        )
    return int((image_length - kernel_length) / stride + 1)


def compute_same_padding_size(kernel_length: int) -> (int, int):
    """
    Computes padding for 'same' padding *per* dimension given

    :param kernel_length: int Kernel size
    :returns: Tuple -- (padStart, padStop) , padding on left/right or top/bottom
    
    Note : Only works for stride 1
    """
    start = kernel_length // 2
    stop = start - 1 if kernel_length % 2 == 0 else start
    return start, stop


def compute_padding(
    kernel_shape: tuple, input_shape: tuple, mode="valid"
) -> (int, int, int, int):
    """
    Computes padding for 'same' or 'valid' padding

    :param kernel_shape: Kernel shape (height, width)
    :param input_shape: Input shape (channels, height, width)
    :param mode: `valid` or `same`
    :return: Tuple -- (pad_left, pad_right, pad_top, pad_bottom)
    """
    if mode == "valid":
        padding = (0, 0, 0, 0)
    elif mode == "same":
        padding = (
            *compute_same_padding_size(kernel_shape[1]),
            *compute_same_padding_size(kernel_shape[0]),
        )
    else:
        raise NotImplementedError("Unknown padding mode")
    return padding


def infer_output_shape(torch_layer, input_shape: Tuple) -> Tuple:
    """
    Compute the output dimensions given input dimensions

    :param torch_layer: a Torch layer
    :param input_shape: the shape of the input tensor
    :return: Tuple -- the size of output tensor
    """
    import torch

    with torch.no_grad():
        # Gen random input
        tsr = torch.rand((1, *input_shape))
        tsrOut = torch_layer(tsr)
        output_shape = tuple(tsrOut.shape)[1:]

    return output_shape
