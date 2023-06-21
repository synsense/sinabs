import torch.nn as nn

class Merge(nn.Module):
    def __init__(self) -> None:
        """
        Module form for a merge operation.
        In the context of events/spikes, events/spikes from two different sources/rasters will be added.
        """
        super().__init__()

    def forward(self, data1, data2):
        size1 = data1.shape
        size2 = data2.shape
        if size1 == size2:
            return data1 + data2
        # If the sizes are not the same, find the larger size and pad the data accordingly
        assert len(size1) == len(size2)
        pad1 = ()
        pad2 = ()
        # Find the larger sizes
        for s1, s2 in zip(size1, size2):
            s_max = max(s1, s2)
            pad1 = (0, s_max-s1, *pad1)
            pad2 = (0, s_max-s2, *pad2)
        
        data1 = nn.functional.pad(input=data1, pad=pad1, mode="constant", value=0)
        data2 = nn.functional.pad(input=data2, pad=pad2, mode="constant", value=0)
        return data1 + data2
