import torch.nn as nn

from sinabs.activation import Quantize


class QuantizeLayer(nn.Module):
    """Layer that quantizes the input, i.e. returns floor(input).

    Parameters:
        quantize: If False, this layer will pass on the input without modifying it.
    """

    def __init__(self, quantize=True):
        super().__init__()
        self.quantize = quantize

    def forward(self, data):
        if self.quantize:
            return Quantize.apply(data)
        else:
            return data
