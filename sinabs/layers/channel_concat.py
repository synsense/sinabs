import torch
import torch.nn as nn

class ConcatenateChannel(nn.Module):
    def __init__(self, channel_axis=-3) -> None:
        super().__init__()
        self.channel_axis = -3

    def forward(self, x, y):
        return torch.cat((x, y), self.channel_axis)