from abc import ABC, abstractmethod
import torch


class SpikingLayer(torch.nn.Module, ABC):
    """
    Pytorch implementation of a spiking neuron with learning enabled.
    This class is the base class for any layer that need to implement leaky or
    non-leaky integrate-and-fire operations.

    See :py:class:`IAF` class for other parameters of this class

    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data):
        pass

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        return dict()
