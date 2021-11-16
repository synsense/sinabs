from typing import Tuple
from warnings import warn
import torch

DEFAULT_STATE_NAME = "state"


class StatefulLayer(torch.nn.Module):
    """
    Pytorch implementation of a stateful layer, to be used as base class.
    """

    backend = "sinabs"

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
        if backend == self.backend:
            return self

        try:
            backend_class = self.get_supported_backends_dict()[backend]
        except KeyError:
            raise RuntimeError(f"Backend '{backend}' not supported.")

        # Generate new instance of corresponding backend class
        new_instance = backend_class(**self._param_dict)

        # Copy parameters
        for name, param in self.named_parameters():
            new_inst_param = getattr(new_instance, name)
            new_inst_param.data = param.data.clone()
        # Copy buffers (using state dict will fail if buffers have non-default shapes)
        for name, buffer in self.named_buffers():
            new_inst_buffer = getattr(new_instance, name)
            new_inst_buffer.data = buffer.data.clone()

        # Warn if parameters of self are not part of new instance
        dropped_params = set(self._param_dict.keys()).difference(
            new_instance._param_dict.keys()
        )
        if dropped_params:
            warn(
                f"Neuron parameter(s) {dropped_params} are not supported in the "
                "target backend and will be ignored.",
                category=RuntimeWarning,
            )

        # Warn if parameters of new instance are not part of self
        new_params = set(new_instance._param_dict.keys()).difference(
            self._param_dict.keys()
        )
        if new_params:
            warn(
                f"Neuron parameter(s) {new_params} in target backend do not exist "
                " in current backend. Default values will be applied.",
                category=RuntimeWarning,
            )
        return new_instance

    def __repr__(self):
        return f"{self.__class__.__name__}-module, backend: {self.backend}"

    @property
    def supported_backends(self) -> Tuple[str]:
        return tuple(self.get_supported_backends_dict.keys())

    @classmethod
    def get_supported_backends_dict(cls):
        if hasattr(cls, "external_backends"):
            return dict({cls.backend: cls}, **cls.external_backends)
        else:
            return {cls.backend: cls}

    def __deepcopy__(self, memo=None):
        copy = self.__class__(**self._param_dict)

        # Copy parameters
        for name, param in self.named_parameters():
            new_inst_param = getattr(copy, name)
            new_inst_param.data = param.data.clone()
        # Copy buffers (using state dict will fail if buffers have non-default shapes)
        for name, buffer in self.named_buffers():
            new_inst_buffer = getattr(copy, name)
            new_inst_buffer.data = buffer.data.clone()  # Copy parameters

        return copy

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        return dict()
