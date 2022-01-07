import torch
from copy import deepcopy
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from .stateful_layer import StatefulLayer
from .squeeze_layer import SqueezeMixin


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
        activation_fn: Callable
            a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
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
    def shape(self):
        if self.is_state_initialised():
            return self.v_mem.shape
        else:
            return None


    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            activation_fn=deepcopy(self.activation_fn),
            threshold_low=self.threshold_low,
            shape = self.shape
        )
        return param_dict


class IAFRecurrent(IAF):
    def __init__(
        self,
        rec_connect: torch.nn.Module,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
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
        rec_out = torch.zeros((batch_size, *trailing_dim), device=input_data.device)
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

    
class IAFSqueeze(IAF, SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width) 
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers. 
    """
    def __init__(self,
                 batch_size = None,
                 num_timesteps = None,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
