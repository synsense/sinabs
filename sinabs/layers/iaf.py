import torch
from copy import deepcopy
from typing import Optional, Callable
from sinabs.activation import MultiSpike, MembraneSubtract, SingleExponential
from .reshape import SqueezeMixin
from .lif import LIF, LIFRecurrent
import numpy as np


class IAF(LIF):
    """
    Integrate and Fire neuron layer.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = V_{mem}(t) + \\sum z(t)

        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    where :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    spike_fn: torch.autograd.Function
        Choose a Sinabs or custom torch.autograd.Function that takes a dict of states,
        a spike threshold and a surrogate gradient function and returns spikes. Be aware
        that the class itself is passed here (because torch.autograd methods are static)
        rather than an object instance.
    reset_fn: Callable
        A function that defines how the membrane potential is reset after a spike.
    surrogate_grad_fn: Callable
        Choose how to define gradients for the spiking non-linearity during the
        backward pass. This is a function of membrane potential.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        spike_threshold: float = 1.0,
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        tau_syn: Optional[float] = None,
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=np.inf,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
            record_states=record_states,
        )
        # IAF does not have time constants
        self.tau_mem = None

    @property
    def alpha_mem_calculated(self):
        return torch.tensor(1.0)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("tau_mem")
        param_dict.pop("train_alphas")
        param_dict.pop("norm_input")
        return param_dict


class IAFRecurrent(LIFRecurrent):
    """
    Integrate and Fire neuron layer with recurrent connections.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = V_{mem}(t) + \\sum z(t)

        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    where :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    rec_connect: torch.nn.Module
        An nn.Module which defines the recurrent connectivity, e.g. nn.Linear
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    spike_fn: torch.autograd.Function
        Choose a Sinabs or custom torch.autograd.Function that takes a dict of states,
        a spike threshold and a surrogate gradient function and returns spikes. Be aware
        that the class itself is passed here (because torch.autograd methods are static)
        rather than an object instance.
    reset_fn: Callable
        A function that defines how the membrane potential is reset after a spike.
    surrogate_grad_fn: Callable
        Choose how to define gradients for the spiking non-linearity during the
        backward pass. This is a function of membrane potential.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        rec_connect: torch.nn.Module,
        spike_threshold: float = 1.0,
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        tau_syn: Optional[float] = None,
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_states: bool = False,
    ):
        super().__init__(
            rec_connect=rec_connect,
            tau_mem=np.inf,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            norm_input=False,
            record_states=record_states,
        )
        # deactivate tau_mem being learned
        self.tau_mem.requires_grad = False

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("tau_mem")
        param_dict.pop("train_alphas")
        param_dict.pop("norm_input")
        return param_dict


class IAFSqueeze(IAF, SqueezeMixin):
    """
    IAF layer with 4-dimensional input (Batch*Time, Channel, Height, Width).

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
