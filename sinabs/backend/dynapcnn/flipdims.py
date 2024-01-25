import torch
import torch.nn as nn
from typing import Tuple


class FlipDims(nn.Module):
    def __init__(
        self, flip_x: bool = False, flip_y: bool = False, swap_xy: bool = False
    ):
        super().__init__()
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy

    def forward(self, data):
        _, _, h, w = list(data.shape)

        # Flip along x and y axis
        if self.flip_y:
            data = data.flip(2)
        if self.flip_x:
            data = data.flip(3)

        if self.swap_xy:
            data = torch.transpose(data, 2, 3)

        return data

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Retuns the output dimensions

        :param input_shape: (channels, height, width)
        :return: (channels, height, width)
        """
        return input_shape
