from typing import Optional, Union

from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class


class LIF(SpikingLayer):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Union[float, None] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron with learning enabled.

        Parameters
        ----------
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: float or None
            Lower bound for membrane potential.
        membrane_subtract: float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        membrane_reset: bool
            If True, reset the membrane to 0 on spiking.
        """
        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )

    def forward(self, data):
        raise NotImplementedError


LIFSqueeze = squeeze_class(LIF)
