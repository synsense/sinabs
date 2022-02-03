from typing import Tuple, List
from warnings import warn
import torch


class StatefulLayer(torch.nn.Module):
    """
    Pytorch implementation of a stateful layer, to be used as base class.
    """

    backend = "sinabs"

    def __init__(self, state_names: List[str]):
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

        for state_name in state_names:
            self.register_buffer(state_name, torch.zeros((0)))

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""
        Zero's the gradients for buffers/state along with the parameters.
        See :meth:`torch.nn.Module.zero_grad` for details
        """
        # Zero grad parameters
        super().zero_grad(set_to_none)
        if self.is_state_initialised():
            # Zero grad buffers
            for b in self.buffers():
                if b.grad_fn is not None:
                    b.detach_()
                else:
                    b.requires_grad_(False)

    def forward(self, *args, **kwargs):
        """
        Not implemented - You need to implement a forward method in child class
        """
        raise NotImplementedError(
            "No forward method has been implemented for this class"
        )

    def is_state_initialised(self) -> bool:
        """ 
        Checks if buffers are of shape 0 and returns 
        True only if none of them are.
        """
        for buffer in self.buffers():
            if buffer.shape == torch.Size([0]):
                return False
        return True
    
    def state_has_shape(self, shape) -> bool:
        """
        Checks if all state have a given shape.
        """
        for buff in self.buffers():
            if buff.shape != shape:
                return False
        return True

    def init_state_with_shape(self, shape, randomize: bool = False) -> None:
        """
        Initialise state/buffers with either zeros or random
        tensor of specific shape.
        """
        for name, buffer in self.named_buffers():
            self.register_buffer(name, torch.zeros(shape, device=buffer.device))
        self.reset_states(randomize=randomize)
        
    def reset_states(self, randomize=False):
        """
        Reset the state/buffers in a layer.
        """
        if self.is_state_initialised():
            for buffer in self.buffers():
                if randomize:
                    torch.nn.init.uniform_(buffer)
                else:
                    buffer.zero_()

    def __repr__(self):
        return f"{self.__class__.__name__}-module, backend: {self.backend}"

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
