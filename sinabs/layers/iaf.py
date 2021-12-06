import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from .stateful_layer import StatefulLayer
from .recurrent_module import recurrent_class
from .pack_dims import squeeze_class


class IAF(StatefulLayer):
    def __init__(
        self,
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
            state_names = ['v_mem']
        )
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low

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


IAFRecurrent = recurrent_class(IAF)
IAFSqueeze = squeeze_class(IAF)
IAFRecurrentSqueeze = squeeze_class(IAFRecurrent)
