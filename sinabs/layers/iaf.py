import torch
from copy import deepcopy
from typing import Optional, Callable
from sinabs.activation import ActivationFunction, activation
from .reshape import SqueezeMixin
from .lif import LIF, LIFRecurrent
from . import functional
import numpy as np


class IAF(LIF):
    """
    Pytorch implementation of a Integrate and Fire neuron with learning enabled.

    Parameters
    ----------
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    activation_fn: Callable
        a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    use_synaptic_state: bool
        Enable / disable synaptic currents. False by default. Note that synaptic currents in this model do not decay.
    """

    def __init__(
        self,
        spike_threshold: float = 1.0,
        activation_fn: Callable = ActivationFunction(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        use_synaptic_state: bool = False,
    ):
        self.use_synaptic_state = use_synaptic_state
        super().__init__(
            tau_mem=np.inf,
            tau_syn=np.inf if use_synaptic_state else None,
            spike_threshold=spike_threshold,
            activation_fn=activation_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("tau_mem")
        param_dict.pop("tau_syn")
        param_dict.pop("train_alphas")
        param_dict.pop("norm_input")
        param_dict["use_synaptic_state"] = self.use_synaptic_state
        return param_dict


class IAFRecurrent(LIFRecurrent):
    """
    Pytorch implementation of a Integrate and Fire neuron with learning enabled.

    Parameters
    ----------
    rec_connect: torch.nn.Module
        An nn.Module which defines the recurrent connectivity, e.g. nn.Linear
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    activation_fn: Callable
        a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    use_synaptic_state: bool
        Enable / disable synaptic currents. False by default. Note that synaptic currents in this model do not decay.
    """

    def __init__(
        self,
        rec_connect: torch.nn.Module,
        spike_threshold: float = 1.0,
        activation_fn: Callable = ActivationFunction(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        use_synaptic_state: bool = False,
    ):
        self.use_synaptic_state = use_synaptic_state
        super().__init__(
            rec_connect=rec_connect,
            tau_mem=np.inf,
            tau_syn=np.inf if use_synaptic_state else None,
            spike_threshold=spike_threshold,
            activation_fn=activation_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("tau_mem")
        param_dict.pop("tau_syn")
        param_dict.pop("train_alphas")
        param_dict.pop("norm_input")
        param_dict["use_synaptic_state"] = self.use_synaptic_state
        return param_dict


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
