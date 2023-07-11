import torch
import torch.nn as nn


class ChannelShift(nn.Module):
    def __init__(self, channel_shift: int = 0, channel_axis=-3) -> None:
        """Given a tensor, shift the channel from the left, ie zero pad from the left.

        Args:
            channel_shift (int, optional): Number of channels to shift by. Defaults to 0.
            channel_axis (int, optional): The channel axis dimension
                NOTE: This has to be a negative dimension such that it counts the dimension from the right. Defaults to -3.
        """
        super().__init__()
        self.padding = []
        self.channel_shift = channel_shift
        self.channel_axis = channel_axis
        for axis in range(-channel_axis):
            self.padding += [0, 0]
        self.padding[-2] = channel_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.pad(input=x, pad=self.padding, mode="constant", value=0)
