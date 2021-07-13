import torch
import torch.nn as nn
from typing import Tuple


class FlipDims(nn.Module):
    def __init__(
            self,
            flip_x: bool = False,
            flip_y: bool = False,
            swap_xy: bool = False
    ):
        super().__init__()
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy

    def forward(self, data):
        _, _, h, w = list(data.shape)
        # Compute indices for X
        index_x = slice(0, w, -1) if self.flip_x else slice(w)

        # Compute indices for y
        index_y = slice(0, h, -1) if self.flip_y else slice(h)

        out = data[:, :, index_y, index_x]

        if self.swap_xy:
            out = torch.swapaxes(out, 2, 3)

        return out

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Retuns the output dimensions

        :param input_shape: (channels, height, width)
        :return: (channels, height, width)
        """
        return input_shape
