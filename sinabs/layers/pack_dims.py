from torch import nn


class SqueezeBatchTime(nn.Module):
    """
    Convenience layer that
    """
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, data):
        all_spikes = data.transpose(0, 1).reshape((-1, *data.shape[2:]))
        return all_spikes


class UnsqueezeBatchTime(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, data):
        syn_out = data.reshape((self.batch_size, -1, *data.shape[1:])).transpose(0, 1)
        return syn_out
