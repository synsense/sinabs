import torch.nn as nn

class LIF(nn.Module):
    def __init__(
            self,
    ):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron.

        """
        super().__init__()

    def forward(self, data):
        raise NotImplementedError