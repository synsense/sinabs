import torch.nn as nn
from typing import Tuple, Optional
from sinabs.layers import Cropping2dLayer
from sinabs.layers import SumPool2d
from .flipdims import FlipDims


class DVSLayer(nn.Module):
    """
    Parameters
    ----------
    input_shape
    pool
    crop
    merge_polarities
    flip_x
    flip_y
    swap_xy
    disable_pixel_array

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
            "swap_xy": swap_xy,
            "flip_x": flip_x,
            "flip_y": flip_y,
        }

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

        #TODO: Account for merge polarity

        channel_count, input_size_y, input_size_x = self.input_shape
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
            "dimensions": self.dimensions,
            "flip": self.flip,
            "crop": self.crop,
            "pooling": self.pooling,
            "disable_pixel_array": self.disable_pixel_array
        }

    def forward(self, data):
        out = self.flip_layer(data)
        if self.crop is not None:
            out = self.crop_layer(out)
        out = self.pool_layer(out)
        return out

    @property
    def pooling(self) -> Tuple[int, int]:
        return self._pooling

    @property
    def crop(self):
        return self._crop

    @property
    def pool_layer(self):
        return self._pool_layer

    @property
    def crop_layer(self):
        return self._crop_layer

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    @property
    def dimensions(self):
        return dict(**self._dimensions)

    @property
    def config_dict(self):
        return dict(**self._config_dict)

    @property
    def flip_layer(self):
        return self._flip_layer

    @property
    def flip(self):
        return dict(**self._flip)
