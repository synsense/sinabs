from typing import Tuple
import torch

DEFAULT_STATE_NAME = "state"


class StatefulLayer(torch.nn.Module):
    """
    Pytorch implementation of a stateful layer, to be used as base class.
    """

    def __init__(self, state_name=None, *args, **kwargs):
        """
        Pytorch implementation of a stateful layer, to be used as base class.

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

        # Blank parameter place holders
        self.register_buffer(state_name or DEFAULT_STATE_NAME, torch.zeros(1))

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

    def forward(self, *args, **kwargs):
        """
        Not implemented - You need to implement a forward method in child class
        """
        raise NotImplementedError(
            "No forward method has been implemented for this class"
        )

    def reset_states(self, shape=None, randomize=False):
        """
        Reset the states in this layer
        """

        device = self.state.device

        for b in self.buffers():
            if shape is None:
                shape = b.shape

            if randomize:
                # State between 0 and 1
                self.state = torch.rand(shape, device=device)
            else:
                self.state = torch.zeros(shape, device=device)

    def to_backend(self, backend):
        try:
            backend_class = self._supported_backends_dict[backend]
        except KeyError:
            raise RuntimeError(f"Backend '{backend}' not supported.")
        else:
            if backend_class == self.__class__:
                return self
            else:
                if backend_class == "slayer":
                    self.cuda()

                # Generate new instance of corresponding backend class
                new_instance = backend_class(**self._param_dict)
                new_instance.load_state_dict(self.state_dict())
                self = new_instance
                return self

    @property
    def supported_backends(self) -> Tuple[str]:
        return tuple(self._supported_backends_dict.keys())

    @property
    def _supported_backends_dict(self) -> dict:
        return {"sinabs": self.__class__}

    def __deepcopy__(self, memo=None):
        copy = self.__class__(**self._param_dict)
        copy.state = self.state.detach().clone()
        copy.activations = self.activations.detach().clone()
        return copy

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        return dict()
