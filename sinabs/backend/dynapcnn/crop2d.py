from typing import Union, List, Tuple
import numpy as np
from torch import nn

ArrayLike = Union[np.ndarray, List, Tuple]


class Crop2d(nn.Module):
    """
    Crop input image by
    """

    def __init__(
            self,
            cropping: ArrayLike = ((0, 0), (0, 0)),
    ):
        """
        Crop input to the the rectangle dimensions

        :param cropping: ((top, bottom), (left, right))
        """
        super().__init__()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        # Crop the data array
        crop_out = binary_input[
                   :,
                   :,
                   self.top_crop: self.bottom_crop,
                   self.left_crop: self.right_crop,
                   ]
        self.out_shape = crop_out.shape[1:]
        self.spikes_number = crop_out.abs().sum()
        self.tw = len(crop_out)
        return crop_out

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Retuns the output dimensions

        :param input_shape: (channels, height, width)
        :return: (channels, height, width)
        """
        channels, height, width = input_shape
        return (
            channels,
            self.bottom_crop - self.top_crop,
            self.right_crop - self.left_crop,
        )

    def __repr__(self):
        return f"Crop2d(({self.top_crop}, {self.bottom_crop}), ({self.left_crop}, {self.right_crop}))"
