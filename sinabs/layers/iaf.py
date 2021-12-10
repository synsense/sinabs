import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from .stateful_layer import StatefulLayer


class IAF(StatefulLayer):
    def __init__(
        self,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
    ):
        """
        Pytorch implementation of a Integrate and Fire neuron with learning enabled.

        Parameters
        ----------
        threshold_low : float or None
            Lower bound for membrane potential.
        """
        super().__init__(
            state_names = ['v_mem']
        )
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        if shape: self.init_state_with_shape(shape)

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass with given data.

        Parameters:
            input_current : torch.Tensor
                Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            torch.Tensor
                Output data. Same shape as `input_data`.
        """

        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape((batch_size, *trailing_dim)):
            self.init_state_with_shape((batch_size, *trailing_dim))

        output_spikes = []
        for step in range(time_steps):
            # Decay the membrane potential and add the input currents which are normalised by tau
            self.v_mem = self.v_mem + input_data[:, step]

            # Clip membrane potential that is too low
            if self.threshold_low:
                self.v_mem = torch.nn.functional.relu(self.v_mem - self.threshold_low) + self.threshold_low

            # generate spikes and adjust v_mem
            spikes, state = self.activation_fn(dict(self.named_buffers()))
            self.v_mem = state['v_mem']

            output_spikes.append(spikes)

        return torch.stack(output_spikes, 1)
    
    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            activation_fn=self.activation_fn,
            shape=self.v_mem.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict


class IAFRecurrent(IAF):
    def __init__(
        self,
        rec_connect: torch.nn.Module,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Integrate and Fire neuron with learning enabled.

        Parameters
        ----------
        threshold_low : float or None
            Lower bound for membrane potential.
        """
        super().__init__(
            activation_fn=activation_fn,
            threshold_low=threshold_low,
        )
        self.rec_connect = rec_connect

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass with given data.

        Parameters:
            input_current : torch.Tensor
                Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            torch.Tensor
                Output data. Same shape as `input_data`.
        """

        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape((batch_size, *trailing_dim)):
            self.init_state_with_shape((batch_size, *trailing_dim))

        output_spikes = []
        rec_out = torch.zeros((batch_size, *trailing_dim))
        for step in range(time_steps):
            total_input = input_data[:, step] + rec_out

            # Decay the membrane potential and add the input currents which are normalised by tau
            self.v_mem = self.v_mem + total_input

            # Clip membrane potential that is too low
            if self.threshold_low:
                self.v_mem = torch.nn.functional.relu(self.v_mem - self.threshold_low) + self.threshold_low

            # generate spikes and adjust v_mem
            spikes, state = self.activation_fn(dict(self.named_buffers()))
            self.v_mem = state['v_mem']

            output_spikes.append(spikes)

            # compute recurrent output that will be added to the input at the next time step
            rec_out = self.rec_connect(spikes).reshape((batch_size, *trailing_dim))

        return torch.stack(output_spikes, 1)

    
class IAFSqueeze(IAF):
    """
    ***Deprecated class, will be removed in future release.***
    """
    def __init__(self,
                 batch_size = None,
                 num_timesteps = None,
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