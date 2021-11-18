from typing import Optional
import torch
from .stateful_layer import StatefulLayer


class SpikingLayer(StatefulLayer):
    """
    Pytorch implementation of a spiking neuron with learning enabled.
    This class is the base class for any layer that need to implement leaky or
    non-leaky integrate-and-fire operations.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        membrane_reset: bool = False,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement leaky or
        non-leaky integrate-and-fire operations.

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
        super().__init__(state_name="v_mem", *args, **kwargs)

        # Initialize neuron states
        self.threshold = threshold
        self.threshold_low = threshold_low
        self._membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset

        # Blank parameter place holders
        self.register_buffer("activations", torch.zeros(1))
        self.spikes_number = None

    def __deepcopy__(self, memo=None):
        # TODO: What is `memo`?
        param_dict = self._param_dict
        other = self.__class__(**param_dict)

        other.v_mem = self.v_mem.detach().clone()
        other.activations = self.activations.detach().clone()

        return other

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        return dict(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self._membrane_subtract,
            membrane_reset=self.membrane_reset,
        )

    @property
    def membrane_subtract(self):
        if self._membrane_subtract is not None:
            return self._membrane_subtract
        else:
            return self.threshold

    @membrane_subtract.setter
    def membrane_subtract(self, new_val):
        self._membrane_subtract = new_val

    def reset_states(self, shape=None, randomize=False):
        """
        Reset the state of all neurons in this layer

        Parameters
        ----------
        shape : None or Tuple of ints
            New shape for states. Generally states can be arbitrary and have no
            time dimension.
        randomize : bool
            If `True`, draw states uniformly between `self.threshold` and
            `self.threshold_low`, or -self.threshold if `self.threshold_low` is
            `None`. Otherwise set states to 0.
        """
        device = self.v_mem.device
        if shape is None:
            shape = self.v_mem.shape

        if randomize:
            # State between lower and upper threshold
            low = self.threshold_low or -self.threshold
            width = self.threshold - low
            self.v_mem = torch.rand(shape, device=device) * width + low
        else:
            self.v_mem = torch.zeros(shape, device=self.v_mem.device)

        self.activations = torch.zeros(shape, device=self.activations.device)
