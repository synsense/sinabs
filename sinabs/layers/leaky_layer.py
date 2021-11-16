from typing import Union
import torch
from .pack_dims import squeeze_class
from .stateful_layer import StatefulLayer


__all__ = ["ExpLeak", "ExpLeakSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ExpLeak(StatefulLayer):
    def __init__(self, alpha: Union[float, torch.Tensor], *args, **kwargs):
        """
        Pytorch implementation of a exponential leaky layer, that is equivalent to an exponential synapse or a low-pass filter.

        Parameters
        ----------
        tau_leak: float
            Rate of leak of the state
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def forward(self, input_current: torch.Tensor):
        # Ensure the neuron state are initialized
        shape_without_time = (input_current.shape[0], *input_current.shape[2:])
        if self.v_mem.shape != shape_without_time:
            self.reset_states(shape=shape_without_time, randomize=False)

        # Determine no. of time steps from input
        time_steps = input_current.shape[1]

        state = self.v_mem

        out_state = []
        for step in range(time_steps):
            state = self.alpha * state  # leak state
            state = state + input_current[:, step]  # Add input
            out_state.append(state)

        self.v_mem = state
        self.tw = time_steps

        out_state = torch.stack(out_state).transpose(0, 1)

        return out_state

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        param_dict = super()._param_dict
        param_dict["alpha"] = self.alpha

        return param_dict


ExpLeakSqueeze = squeeze_class(ExpLeak)
