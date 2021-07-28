from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import torch


class SpikingLayer(torch.nn.Module, ABC):
    """
    Pytorch implementation of a spiking neuron with learning enabled.
    This class is the base class for any layer that need to implement leaky or
    non-leaky integrate-and-fire operations.
    This is an abstract base class.
    """

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
        Pytorch implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement leaky or
        non-leaky integrate-and-fire operations.
        This is an abstract base class.

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
        super().__init__()

        # Initialize neuron states
        self.threshold = threshold
        self.threshold_low = threshold_low
        self._membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset

        # Blank parameter place holders
        self.register_buffer("state", torch.zeros(1))
        self.register_buffer("activations", torch.zeros(1))
        self.spikes_number = None

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""
        Zero's the gradients for buffers/states along with the parameters.
        See :meth:`torch.nn.Module.zero_grad` for details
        """
        # Zero grad parameters
        super().zero_grad(set_to_none)
        # Zero grad buffers
        for b in self.buffers():
            if b.grad_fn is not None:
                b.detach_()
            else:
                b.requires_grad_(False)

    def get_output_shape(self, in_shape: Tuple[int]) -> Tuple[int]:
        """
        Returns the output shape for passthrough implementation

        Parameters
        ----------
        in_shape: Tuple of integers
            Input shape

        Returns
        -------
        Tuple of input shape
            Output shape at given input shape
        """
        return in_shape

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
        device = self.state.device
        if shape is None:
            shape = self.state.shape

        if randomize:
            # State between lower and upper threshold
            low = self.threshold_low or -self.threshold
            width = self.threshold - low
            self.state = torch.rand(shape, device=device) * width + low
        else:
            self.state = torch.zeros(shape, device=self.state.device)

        self.activations = torch.zeros(shape, device=self.activations.device)

    def __deepcopy__(self, memo=None):
        # TODO: What is `memo`?

        other = self.__class__(**self._param_dict)

        other.state = self.state.detach().clone()
        other.activations = self.activations.detach().clone()

        return other

    @abstractmethod
    def forward(self, data):
        pass

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
