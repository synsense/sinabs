from typing import Union, Optional
import torch
import torch.nn as nn
from .stateful_layer import StatefulLayer


class ExpLeak(StatefulLayer):
    def __init__(self, 
                 tau_leak: Union[float, torch.Tensor],
                 shape: Optional[torch.Size] = None,
                 train_alphas: bool = False,
                 threshold_low: Optional[float] = None,
                ):
        """
        Pytorch implementation of a exponential leaky layer, that is equivalent to an exponential synapse or a low-pass filter.

        Parameters
        ----------
        tau: float
            Rate of leak of the state
        """
        super().__init__(
            state_names = ['v_mem']
        )
        if train_alphas:
            self.alpha_leak = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_leak)))
        else:
            self.tau_leak = nn.Parameter(torch.as_tensor(tau_leak))
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_leak_calculated(self):
        return self.alpha_leak if self.train_alphas else torch.exp(-1/self.tau_leak)
    
    def forward(self, input_data: torch.Tensor):
        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape((batch_size, *trailing_dim)):
            self.init_state_with_shape((batch_size, *trailing_dim))

        alpha_leak = self.alpha_leak_calculated
        
        output_states = []
        for step in range(time_steps):
            # Decay membrane potential and add synaptic inputs
            self.v_mem = alpha_leak * self.v_mem + (1 - alpha_leak) * input_data[:, step]

            # Clip membrane potential that is too low
            if self.threshold_low:
                self.v_mem = torch.nn.functional.relu(self.v_mem - self.threshold_low) + self.threshold_low

            output_states.append(self.v_mem)

        return torch.stack(output_states, 1)
    
    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            tau_leak=-1/torch.log(self.alpha_leak.detach_()) if self.train_alphas else self.tau_leak,
            train_alphas=self.train_alphas,
            shape=self.v_mem.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict


class ExpLeakSqueeze(ExpLeak):
    """
    ***Deprecated class, will be removed in future release.***
    """
    def __init__(self,
                 batch_size = None,
                 num_timesteps = None,
                 *args,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        if not batch_size and not num_timesteps:
            raise TypeError("You need to specify either batch_size or num_timesteps.")
        if not batch_size:
            batch_size = -1 
        if not num_timesteps:
            num_timesteps = -1
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        inflated_input = input_data.reshape(self.batch_size, self.num_timesteps, *input_data.shape[1:])
        
        inflated_output = super().forward(inflated_input)
        
        return inflated_output.flatten(start_dim=0, end_dim=1)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
        )
        return param_dict