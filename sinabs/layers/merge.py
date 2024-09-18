import torch.nn as nn


class Merge(nn.Module):
    def __init__(self, num_inputs=None) -> None:
        """Module form for a merge operation.

        In the context of events/spikes, events/spikes from two different sources/rasters will be
        added.
        """
        super().__init__()
        self.num_inputs = num_inputs

    def forward(self, *data):
        sizes = [x.shape for x in data]

        if self.num_inputs is not None and len(data) != self.num_inputs:
            raise ValueError(
                f"This Merge layer expects exactly {self.num_inputs}, "
                f"but received {len(data)}"
            )

        if len(sizes) == 0:
            # No data provided, return None
            raise ValueError("Number of inputs cannot be 0")

        if all(s == sizes[0] for s in sizes):
            # All sizes are the same and can be added
            return sum(sizes)

        # If the sizes are not the same, find the largest size and pad the data accordingly
        num_dims = len(sizes[0])
        if not all(len(s) == num_dims for s in sizes):
            raise ValueError("All inputs must have the same number of dimensions")
        # Find largest size in each dimension
        max_size = (max(s[dim] for s in sizes) for dim in range(num_dims))
        # Determine padding for each input and dimension
        # Each nested tuple will correspond to one input and be of the
        # form (0, p0, 0, p1, ...) to pad only in the end.
        padding: Tuple[Tuple[int]] = (
            (p for dim in range(num_dims) for p in (0, max_size[dim] - s[dim]))
            for s in sizes
        )
        padded_data = (
            nn.functional.pad(input=x, pad=pad, mode="constant", value=0)
            for x, pad in zip(data, padding)
        )
        return sum(padded_data)
