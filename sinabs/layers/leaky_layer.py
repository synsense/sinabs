from typing import Tuple, Union
import torch
from .pack_dims import squeeze_class


__all__ = ["ExpLeak", "ExpLeakSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ExpLeak(torch.nn.Module):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
    ):
        """
        Pytorch implementation of a exponential leaky layer, that is equivalent to a exponential synapse or a low-pass filter.

        Parameters
        ----------
        tau_leak: float
            Rate of leak of the state
        """
        super().__init__()
        self.register_buffer("state", torch.zeros(1))
        self.tau_mem = tau_mem

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

    def forward(self, input_current: torch.Tensor):
        # Ensure the neuron state are initialized
        shape_notime = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_notime:
            self.reset_states(shape=shape_notime, randomize=False)

        # Determine no. of time steps from input
        time_steps = input_current.shape[1]

        state = self.state

        alpha = torch.exp(- 1.0/self.tau_mem)

        out_state = []

        for iCurrentTimeStep in range(time_steps):
            state = alpha * state  # leak state
            state = state + input_current[:, iCurrentTimeStep]  # Add input
            out_state.append(state)

        self.state = state
        self.tw = time_steps

        out_state = torch.stack(out_state).transpose(0, 1)

        return out_state


ExpLeakSqueeze = squeeze_class(ExpLeak)
