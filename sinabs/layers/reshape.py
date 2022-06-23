import torch
import torch.nn as nn
from typing import Callable, Optional


class FlattenTime(nn.Flatten):
    """
    Utility layer which always flattens first two dimensions. Meant
    to convert a tensor of dimensions (Batch, Time, Channels, Height, Width)
    into a tensor of (Batch*Time, Channels, Height, Width).
    """

    def __init__(self):
        super().__init__(start_dim=0, end_dim=1)


class UnflattenTime(nn.Module):
    """
    Utility layer which always unflattens (expands) the first dimension into two separate ones.
    Meant to convert a tensor of dimensions (Batch*Time, Channels, Height, Width)
    into a tensor of (Batch, Time, Channels, Height, Width).
    """

    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, x):
        num_time_steps = x.shape[0] // self.batch_size
        return x.unflatten(0, (self.batch_size, num_time_steps))


class SqueezeMixin:
    """
    Utility mixin class that will wrap the __init__ and forward call of other classes.
    The wrapped __init__ will provide two additional parameters batch_size and num_timesteps
    and the wrapped forward will unpack and repack the first dimension into batch and time.
    """

    def squeeze_init(self, batch_size: Optional[int], num_timesteps: Optional[int]):
        if not batch_size and not num_timesteps:
            raise TypeError("You need to specify either batch_size or num_timesteps.")
        if not batch_size:
            batch_size = -1
        if not num_timesteps:
            num_timesteps = -1
        self.batch_size = int(batch_size)
        self.num_timesteps = int(num_timesteps)

    def squeeze_forward(self, input_data: torch.Tensor, forward_method: Callable):
        inflated_input = input_data.reshape(
            self.batch_size, self.num_timesteps, *input_data.shape[1:]
        )
        inflated_output = forward_method(inflated_input)
        return inflated_output.flatten(start_dim=0, end_dim=1)

    def squeeze_param_dict(self, param_dict: dict) -> dict:
        param_dict.update(
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
        )
        return param_dict
