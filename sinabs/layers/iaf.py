import torch
from copy import deepcopy
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from .stateful_layer import StatefulLayer
from .squeeze_layer import SqueezeMixin
from . import functional


class IAF(StatefulLayer):
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

    def __init__(
        self,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
    ):
        super().__init__(state_names=["v_mem"])
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        if shape:
            self.init_state_with_shape(shape)

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
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        spikes, state = functional.lif_forward(
            input_data=input_data,
            alpha_mem=1.0,
            alpha_syn=None,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            threshold_low=self.threshold_low,
            norm_input=False,
        )
        self.v_mem = state["v_mem"]

        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes

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
            shape=self.shape,
        )
        return param_dict


class IAFRecurrent(IAF):
    """
    Pytorch implementation of a Integrate and Fire neuron with learning enabled.

    Parameters
    ----------
    threshold_low : float or None
        Lower bound for membrane potential.
    """

    def __init__(
        self,
        rec_connect: torch.nn.Module,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
    ):
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
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        spikes, state = functional.lif_recurrent(
            input_data=input_data,
            alpha_mem=1.0,
            alpha_syn=None,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            threshold_low=self.threshold_low,
            norm_input=False,
            rec_connect=self.rec_connect,
        )
        self.v_mem = state["v_mem"]

        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes


class IAFSqueeze(IAF, SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.
    """

    def __init__(
        self,
        batch_size=None,
        num_timesteps=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
