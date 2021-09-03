from typing import Tuple, Union
import torch
from .pack_dims import squeeze_class
from .spiking_layer import SpikingLayer


__all__ = ["ExpLeak", "ExpLeakSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ExpLeak(SpikingLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a exponential leaky layer, that is equivalent to a exponential synapse or a low-pass filter.

        Parameters
        ----------
        tau_leak: float
            Rate of leak of the state
        """
        super().__init__(*args, **kwargs)
        self.register_buffer("state", torch.zeros(1))
        self.tau_mem = tau_mem

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
