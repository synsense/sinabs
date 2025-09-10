from typing import List, Tuple, Union

import numpy as np
from torch import nn

ArrayLike = Union[np.ndarray, List, Tuple]


class Cropping2dLayer(nn.Module):
    """Crop input image by.

    Args:
        cropping: ((top, bottom), (left, right))
    """

    def __init__(
        self,
        cropping: ArrayLike = ((0, 0), (0, 0)),
    ):
        super().__init__()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        _, self.channels_in, h, w = list(binary_input.shape)
        # Crop the data array
        crop_out = binary_input[
            :,
            :,
            self.top_crop : h - self.bottom_crop,
            self.left_crop : w - self.right_crop,
        ]
        self.out_shape = crop_out.shape[1:]
        self.spikes_number = crop_out.abs().sum()
        self.tw = len(crop_out)
        return crop_out

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """Retuns the output dimensions.

        Args:
            input_shape: (channels, height, width)

        Returns:
            (channels, height, width)
        """
        channels, height, width = input_shape
        return (
            channels,
            height - self.top_crop - self.bottom_crop,
            width - self.left_crop - self.right_crop,
        )
