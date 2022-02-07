import torch.nn as nn


class FlattenTime(nn.Flatten):
    """
    Utility layer which always flattens first two dimensions. Meant
    to convert a tensor of dimensions (Batch, Time, Channels, Height, Width)
    into a tensor of (Batch*Time, Channels, Height, Width).
    """

    def __init__(self):
        super().__init__(start_dim=0, end_dim=1)


class UnflattenTime(nn.Unflatten):
    """
    Utility layer which always unflattens (expands) the first dimension into two separate ones.
    Meant to convert a tensor of dimensions (Batch*Time, Channels, Height, Width)
    into a tensor of (Batch, Time, Channels, Height, Width).
    """

    def __init__(self, batch_size: int):
        super().__init__(dim=0, unflattened_size=(batch_size, -1))
