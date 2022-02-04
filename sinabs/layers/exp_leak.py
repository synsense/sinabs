from typing import Union, Optional
import torch
import torch.nn as nn
from .stateful_layer import StatefulLayer
from .squeeze_layer import SqueezeMixin


class ExpLeak(StatefulLayer):
    """
    Pytorch implementation of a exponential leaky layer, that is equivalent to an exponential synapse or a low-pass filter.

    Parameters
    ----------
    tau: float
        Rate of leak of the state
    """
    def __init__(
        self,
        tau_leak: Union[float, torch.Tensor],
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        threshold_low: Optional[float] = None,
    ):
        super().__init__(state_names=["v_mem"])
        tau_leak = torch.as_tensor(tau_leak, dtype=float)
        if train_alphas:
            self.alpha_leak = nn.Parameter(torch.exp(-1 / tau_leak))
        else:
            self.tau_leak = nn.Parameter(tau_leak)
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_leak_calculated(self):
        return self.alpha_leak if self.train_alphas else torch.exp(-1 / self.tau_leak)

    def forward(self, input_data: torch.Tensor):
        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        alpha_leak = self.alpha_leak_calculated

        output_states = []
        for step in range(time_steps):
            # Decay membrane potential and add synaptic inputs
            self.v_mem = (
                alpha_leak * self.v_mem + (1 - alpha_leak) * input_data[:, step]
            )

            # Clip membrane potential that is too low
            if self.threshold_low:
                self.v_mem = (
                    torch.nn.functional.relu(self.v_mem - self.threshold_low)
                    + self.threshold_low
                )

            output_states.append(self.v_mem)

        return torch.stack(output_states, 1)

    @property
    def shape(self):
        if self.is_state_initialised():
            return self.v_mem.shape
        else:
            return None

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            tau_leak=-1 / torch.log(self.alpha_leak.detach_())
            if self.train_alphas
            else self.tau_leak,
            train_alphas=self.train_alphas,
            shape=self.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict


class ExpLeakSqueeze(ExpLeak, SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width) 
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers. 
    """
    def __init__(self, batch_size=None, num_timesteps=None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
