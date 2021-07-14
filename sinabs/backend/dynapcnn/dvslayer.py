import torch.nn as nn
from typing import Tuple, Optional
from sinabs.layers import Cropping2dLayer
from sinabs.layers import SumPool2d
from .flipdims import FlipDims


class DVSLayer(nn.Module):
    """
    DVSLayer representing the DVS pixel array on chip and/or the pre-processing.

    Parameters
    ----------
    input_shape;
        Shape of input (height, width)
    pool:
        Sum pooling kernel size (height, width)
    crop:
        Crop the input to the given ROI ((top, bottom), (left, right))
    merge_polarities:
        If true, events from both polarities will be merged.
    flip_x:
        Flip the X axis
    flip_y:
        Flip the Y axis
    swap_xy:
        Swap X and Y dimensions
    disable_pixel_array:
        Disable the pixel array. This is useful if you want to use the DVS layer for input preprocessing.
    """

    def __init__(
            self,
            input_shape: Tuple[int, int],
            pool: Tuple[int, int] = (1, 1),
            crop: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
            merge_polarities: bool = False,
            flip_x: bool = False,
            flip_y: bool = False,
            swap_xy: bool = False,
            disable_pixel_array: bool = True,
    ):
        super().__init__()

        # Initialize crop layer
        if crop is not None:
            self._crop_layer = Cropping2dLayer(crop)
            self._crop = {
                "origin": {"x": crop[1][0], "y": crop[0][0]},
                "cut": {"x": crop[1][1], "y": crop[0][1]},
            }
        else:
            self._crop_layer = None
            self._crop = None

        # Initialize flip layer
        self._flip_layer = FlipDims(flip_x, flip_y, swap_xy)
        self._flip = {
            "flip_x": flip_x,
            "flip_y": flip_y,
        }
        self._swap_xy = swap_xy

        # Initialize pooling layer
        self._pooling = pool
        self._pool_layer = SumPool2d(pool)

        self._config_dict = {}
        self._input_shape = input_shape
        self._update_config_dict()

        # DVS specific settings
        self.merge_polarities = merge_polarities
        self.disable_pixel_array = disable_pixel_array

    def _update_dimensions(self):
        channel_count, input_size_y, input_size_x = self.input_shape

        if self.merge_polarities:
            channel_count = 1

        input_shape = {
            "size": {"x": input_size_x, "y": input_size_y},
            "feature_count": channel_count,
        }
        # Compute dims after cropping
        if self.crop is not None:
            channel_count, input_size_y, input_size_x = self.crop_layer.get_output_shape(self.input_shape)

        # Compute dims after pooling
        output_shape = {
            "size": {
                "x": input_size_x // self.pooling[1],
                "y": input_size_y // self.pooling[0],
            },
            "feature_count": channel_count,
        }
        self._dimensions = {"input_shape": input_shape, "output_shape": output_shape}
        self._output_shape = output_shape

    def _update_config_dict(self):
        self._update_dimensions()
        self._config_dict = {
            "merge": self.merge_polarities,
            "dimensions": self.dimensions,
            "mirror": self.flip,
            "mirror_diagonal": self.swap_xy,
            "crop": self.crop,
            "pooling": self.pooling,
            "pass_sensor_events": not self.disable_pixel_array
        }

    def forward(self, data):
        if self.merge_polarities:
            data = data.sum(1, keepdim=True)

        if self.crop is not None:
            out = self.crop_layer(data)
        else:
            out = data
        out = self.flip_layer(out)
        out = self.pool_layer(out)
        return out

    @property
    def pooling(self) -> Tuple[int, int]:
        return self._pooling

    @property
    def crop(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self._crop

    @property
    def output_shape(self) -> dict:
        return self._output_shape

    @property
    def flip(self) -> dict:
        return dict(**self._flip)

    @property
    def swap_xy(self) -> bool:
        return self._swap_xy

    @property
    def dimensions(self) -> dict:
        return dict(**self._dimensions)

    @property
    def config_dict(self) -> dict:
        return dict(**self._config_dict)

    @property
    def pool_layer(self) -> nn.Module:
        return self._pool_layer

    @property
    def crop_layer(self) -> nn.Module:
        return self._crop_layer

    @property
    def flip_layer(self) -> nn.Module:
        return self._flip_layer
