import torch.nn as nn
from typing import Tuple, Optional
from sinabs.layers import SumPool2d
from .flipdims import FlipDims
from .crop2d import Crop2d


def expand_to_pair(value) -> (int, int):
    """
    Expand a given value to a pair (tuple) if an int is passed

    Parameters
    ----------
    value:
        int

    Returns
    -------
    pair:
        (int, int)
    """
    return (value, value) if isinstance(value, int) else value


class DVSLayer(nn.Module):
    """
    DVSLayer representing the DVS pixel array on chip and/or the pre-processing.
    The order of processing is as follows
    MergePolarity -> Pool -> Cut -> Flip

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

        # DVS specific settings
        self.merge_polarities = merge_polarities
        self.disable_pixel_array = disable_pixel_array

        if len(input_shape) != 2:
            raise ValueError(f"Input shape should be 2 dimensional but input_shape={input_shape} was given.")
        if merge_polarities:
            self.input_shape: Tuple[int, int, int] = (1, *input_shape)
        else:
            self.input_shape: Tuple[int, int, int] = (2, *input_shape)

        # Initialize pooling layer
        self.pool_layer = SumPool2d(pool)

        # Initialize crop layer
        if crop is None:
            num_channels, height, width = self.get_output_shape_after_pooling()
            crop = ((0, height), (0, width))
        self.crop_layer = Crop2d(crop)

        # Initialize flip layer
        self.flip_layer = FlipDims(flip_x, flip_y, swap_xy)

    @classmethod
    def from_layers(
            cls,
            input_shape: Tuple[int, int, int],
            pool_layer: Optional[SumPool2d] = None,
            crop_layer: Optional[Crop2d] = None,
            flip_layer: Optional[FlipDims] = None,
    ) -> "DVSLayer":
        """
        Alternative factory method.
        Generate a DVSLayer from a set of torch layers

        Parameters
        ----------
        input_shape:
            (channels, height, width)
        pool_layer:
            SumPool2d layer
        crop_layer:
            Crop2d layer
        flip_layer:
            FlipDims layer

        Returns
        -------
        DVSLayer

        """
        pool = (1, 1)
        crop = None
        flip_x = None
        flip_y = None
        swap_xy = None

        if len(input_shape) != 3:
            raise ValueError(f"Input shape should be 3 dimensional but input_shape={input_shape} was given.")

        if pool_layer is not None:
            pool = expand_to_pair(pool_layer.kernel_size)
        if crop_layer is not None:
            crop = (
                (crop_layer.top_crop, crop_layer.bottom_crop),
                (crop_layer.left_crop, crop_layer.right_crop),
            )
        if flip_layer is not None:
            flip_x = flip_layer.flip_x
            flip_y = flip_layer.flip_y
            swap_xy = flip_layer.swap_xy

        return DVSLayer(
            input_shape=input_shape[1:],
            pool=pool,
            crop=crop,
            flip_x=False if flip_x is None else flip_x,
            flip_y=False if flip_y is None else flip_y,
            swap_xy=False if swap_xy is None else swap_xy,
            merge_polarities=(input_shape[0] == 1)
        )

    @property
    def input_shape_dict(self) -> dict:
        """
        The configuration dictionary for the input shape

        Returns
        -------
        dict
        """
        channel_count, input_size_y, input_size_x = self.input_shape

        if self.merge_polarities:
            channel_count = 1

        return {
            "size": {"x": input_size_x, "y": input_size_y},
            "feature_count": channel_count,
        }

    def get_output_shape_after_pooling(self) -> Tuple[int, int, int]:
        """
        Get the shape of data just after the pooling layer.

        Returns
        -------
        (channel, height, width)
        """
        channel_count, input_size_y, input_size_x = self.input_shape

        if self.merge_polarities:
            channel_count = 1

        # Compute shapes after pooling
        pooling = self.get_pooling()
        output_size_x = input_size_x // pooling[1]
        output_size_y = input_size_y // pooling[0]
        return channel_count, output_size_y, output_size_x

    def get_output_shape_dict(self) -> dict:
        """
        Configuration dictionary for output shape

        Returns
        -------
        dict
        """
        channel_count, output_size_y, output_size_x = (
            self.get_output_shape_after_pooling()
        )

        # Compute dims after cropping
        if self.crop_layer is not None:
            channel_count, output_size_y, output_size_x = self.crop_layer.get_output_shape(
                (channel_count, output_size_y, output_size_x)
            )

        # Compute dims after pooling
        return {
            "size": {"x": output_size_x, "y": output_size_y},
            "feature_count": channel_count,
        }

    def get_config_dict(self) -> dict:
        crop = self.get_roi()
        cut = {"x": crop[1][1] - 1, "y": crop[0][1] - 1}
        origin = {"x": crop[1][0], "y": crop[0][0]}
        pooling = {"y": self.get_pooling()[0], "x": self.get_pooling()[1]}

        return {
            "merge": self.merge_polarities,
            "mirror": self.get_flip_dict(),
            "mirror_diagonal": self.get_swap_xy(),
            "cut": cut,
            "origin": origin,
            "pooling": pooling,
            "pass_sensor_events": not self.disable_pixel_array,
        }

    def forward(self, data):
        # Merge polarities
        if self.merge_polarities:
            data = data.sum(1, keepdim=True)
        # Pool
        out = self.pool_layer(data)
        # Crop
        if self.crop_layer is not None:
            out = self.crop_layer(out)
        # Flip stuff
        out = self.flip_layer(out)

        return out

    def get_pooling(self) -> Tuple[int, int]:
        """
        Pooling kernel shape

        Returns
        -------
        (ky, kx)
        """
        return expand_to_pair(self.pool_layer.kernel_size)

    def get_roi(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        The coordinates for ROI.
        Note that this is not the same as crop parameter passed during the object construction.

        Returns
        -------
        ((top, bottom), (left, right))
        """
        if self.crop_layer is not None:
            _, h, w = self.get_output_shape_after_pooling()
            return (
                (self.crop_layer.top_crop, self.crop_layer.bottom_crop),
                (self.crop_layer.left_crop, self.crop_layer.right_crop),
            )
        else:
            _, output_size_y, output_size_x = self.get_output_shape()
            return (0, output_size_y), (0, output_size_x)

    def get_output_shape(self) -> Tuple[int, int, int]:
        """
        Output shape of the layer

        Returns
        -------
        (channel, height, width)
        """
        channel_count, input_size_y, input_size_x = self.input_shape

        if self.merge_polarities:
            channel_count = 1

        # Compute shapes after pooling
        pooling = self.get_pooling()
        output_size_x = input_size_x // pooling[1]
        output_size_y = input_size_y // pooling[0]

        # Compute dims after cropping
        if self.crop_layer is not None:
            channel_count, output_size_y, output_size_x = self.crop_layer.get_output_shape(
                (channel_count, output_size_y, output_size_x)
            )

        return channel_count, output_size_y, output_size_x

    def get_flip_dict(self) -> dict:
        """
        Configuration dictionary for x, y flip

        Returns
        -------
        dict
        """

        return {"x": self.flip_layer.flip_x, "y": self.flip_layer.flip_y}

    def get_swap_xy(self) -> bool:
        """
        True if XY has to be swapped.

        Returns
        -------
        bool
        """
        return self.flip_layer.swap_xy
