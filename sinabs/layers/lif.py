from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class


class LIF(SpikingLayer):
    def __init__(self,):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron.

        """
        super().__init__()

    def forward(self, data):
        raise NotImplementedError


LIFSqueeze = squeeze_class(LIF)
