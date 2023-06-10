import torch.nn as nn

class Add(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """
        Module form for a simple addition operation.
        In the context of events/spikes, events/spikes from two different sources/rasters will be added.
        """
        super().__init__(*args, **kwargs)

    def forward(self, data1, data2):
        return data1 + data2