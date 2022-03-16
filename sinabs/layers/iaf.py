import torch
from copy import deepcopy
from typing import Optional, Callable
from sinabs.activation import ActivationFunction, activation
from .stateful_layer import StatefulLayer
from .reshape import SqueezeMixin
from .lif import LIF
from . import functional
import numpy as np


class IAF(LIF):
    """
    Pytorch implementation of a Integrate and Fire neuron with learning enabled.

    Parameters
    ----------
    activation_fn: Callable
        a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    """

    def __init__(
        self,
        activation_fn: Callable = ActivationFunction(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        use_synaptic_state: bool = False,
    ):
        self.use_synaptic_state = use_synaptic_state
        super().__init__(
            tau_mem=np.inf,
            tau_syn=np.inf if use_synaptic_state else None,
            activation_fn=activation_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop('tau_mem')
        param_dict.pop('tau_syn')
        param_dict.pop('train_alphas')
        param_dict.pop('norm_input')
        param_dict['use_synaptic_state'] = self.use_synaptic_state
        return param_dict


class IAFRecurrent(IAF):
    """
    Pytorch implementation of a Integrate and Fire neuron with learning enabled.

    Parameters
    ----------
    min_v_mem : float or None
        Lower bound for membrane potential.
    """

    def __init__(
        self,
        rec_connect: torch.nn.Module,
        activation_fn: Callable = ActivationFunction(),
        min_v_mem: Optional[float] = None,
    ):
        super().__init__(
            activation_fn=activation_fn,
            min_v_mem=min_v_mem,
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
            min_v_mem=self.min_v_mem,
            norm_input=False,
            rec_connect=self.rec_connect,
        )
        self.v_mem = state["v_mem"]

        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes


class IAFSqueeze(IAF, SqueezeMixin):
    """
    Same as parent IAF class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
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
