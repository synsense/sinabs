from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential

from . import functional
from .reshape import SqueezeMixin
from .stateful_layer import StatefulLayer


class LIF(StatefulLayer):
    """Leaky Integrate and Fire neuron layer that inherits from
    :class:`~sinabs.layers.StatefulLayer`.

    Neuron dynamics in discrete time for norm_input=True:

    .. math ::
        V_{mem}(t+1) = max(\\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t), V_{min})

    Neuron dynamics for norm_input=False:

    .. math ::
        V_{mem}(t+1) = max(\\alpha V_{mem}(t) + \\sum z(t), V_{min})

    where :math:`\\alpha =  e^{-1/tau_{mem}}`, :math:`V_{min}` is a minimum membrane potential
    and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.
    We also reset the membrane potential according to reset_fn:

    .. math ::
        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    Parameters:
        tau_mem: Membrane potential time constant.
        tau_syn: Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
        spike_threshold: Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
        spike_fn: Choose a Sinabs or custom torch.autograd.Function that takes a dict of states,
            a spike threshold and a surrogate gradient function and returns spikes. Be aware
            that the class itself is passed here (because torch.autograd methods are static)
            rather than an object instance.
        reset_fn: Specify how a neuron's membrane potential should be reset after a spike.
        surrogate_grad_fn: Choose how to define gradients for the spiking non-linearity during the
            backward pass. This is a function of membrane potential.
        min_v_mem: Lower bound for membrane potential v_mem, clipped at every time step.
        train_alphas: When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
        shape: Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        norm_input: When True, normalise input current by tau. This helps when training time constants.
        record_states: When True, will record all internal states such as v_mem or i_syn in a dictionary
            attribute `recordings`. Default is False.

    Shape:
        - Input: :math:`(Batch, Time, Channel, Height, Width)` or :math:`(Batch, Time, Channel)`
        - Output: Same as input.

    Attributes:
        v_mem: The membrane potential resets according to reset_fn for every spike.
        i_syn: This attribute is only available if tau_syn is not None.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            state_names=["v_mem", "i_syn"] if tau_syn is not None else ["v_mem"]
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(
                torch.exp(-1.0 / torch.as_tensor(tau_mem, dtype=torch.float32))
            )
            self.alpha_syn = (
                nn.Parameter(
                    torch.exp(-1.0 / torch.as_tensor(tau_syn, dtype=torch.float32))
                )
                if tau_syn is not None
                else None
            )
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem, dtype=torch.float32))
            self.tau_syn = (
                nn.Parameter(torch.as_tensor(tau_syn, dtype=torch.float32))
                if tau_syn is not None
                else None
            )
        self.spike_fn = spike_fn
        self.reset_fn = reset_fn
        self.surrogate_grad_fn = surrogate_grad_fn
        self.train_alphas = train_alphas
        self.norm_input = norm_input
        self.record_states = record_states
        self.min_v_mem = (
            nn.Parameter(torch.as_tensor(min_v_mem), requires_grad=False)
            if min_v_mem is not None
            else None
        )
        self.spike_threshold = (
            nn.Parameter(torch.as_tensor(spike_threshold), requires_grad=False)
            if spike_threshold is not None
            else None
        )
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self) -> torch.Tensor:
        """Calculates alpha_mem from tau_mem, if not already known."""
        if self.train_alphas:
            return self.alpha_mem
        else:
            return torch.exp(-1.0 / self.tau_mem)

    @property
    def alpha_syn_calculated(self) -> torch.Tensor:
        """Calculates alpha_syn from tau_syn, if not already known."""
        if self.train_alphas:
            return self.alpha_syn
        elif self.tau_syn is not None:
            # Calculate alpha with 64 bit floating point precision
            return torch.exp(-1.0 / self.tau_syn)
        else:
            return None

    @property
    def tau_mem_calculated(self) -> torch.Tensor:
        """Calculates tau_mem from alpha_mem, if not already known."""
        if self.train_alphas:
            if self.alpha_mem is None:
                return None
            else:
                return -1.0 / torch.log(self.alpha_mem)
        else:
            return self.tau_mem

    @property
    def tau_syn_calculated(self) -> torch.Tensor:
        """Calculates tau_syn from alpha_syn, if not already known."""
        if self.train_alphas:
            if self.alpha_syn is None:
                return None
            else:
                return -1.0 / torch.log(self.alpha_syn)
        else:
            return self.tau_syn

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_data: Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            Output data with same shape as `input_data`.
        """
        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the state is initialized.
        if not self.is_state_initialised():
            self.init_state_with_shape((batch_size, *trailing_dim))
        
        if not self.state_has_shape((batch_size, *trailing_dim)):
            # If the trailing dim has changed, we reinitialize the states.
            if not self.has_trailing_dimension(trailing_dim):
                self.init_state_with_shape((batch_size, *trailing_dim))
            # Otherwise only the batch size has changed.
            else: 
                self.handle_state_batch_size_mismatch(batch_size) # with the input batch size


        alpha_mem = self.alpha_mem_calculated
        alpha_syn = self.alpha_syn_calculated

        spikes, state, recordings = functional.lif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            spike_threshold=self.spike_threshold,
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )
        self.v_mem = state["v_mem"]
        self.i_syn = state["i_syn"] if alpha_syn is not None else None
        self.recordings = recordings

        self.firing_rate = spikes.mean()
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
        tau_syn = self.tau_syn_calculated
        if tau_syn is not None:
            tau_syn = tau_syn.detach()
        tau_mem = self.tau_mem_calculated
        if tau_mem is not None:
            tau_mem = tau_mem.detach()

        param_dict.update(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            spike_threshold=self.spike_threshold,
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            train_alphas=self.train_alphas,
            shape=self.shape,
            min_v_mem=self.min_v_mem,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )
        return param_dict


class LIFRecurrent(LIF):
    """Leaky Integrate and Fire neuron layer with recurrent connections which inherits from
    :class:`~sinabs.layers.LIF`.

    Neuron dynamics in discrete time for norm_input=True:

    .. math ::
        V_{mem}(t+1) = max(\\alpha V_{mem}(t) + (1-\\alpha)\\sum z_{in}(t) z_{rec}(t), V_{min})

    Neuron dynamics for norm_input=False:

    .. math ::
        V_{mem}(t+1) = max(\\alpha V_{mem}(t) + \\sum z_{in}(t) z_{rec}(t), V_{min})

    where :math:`\\alpha =  e^{-1/tau_{mem}}`, :math:`V_{min}` is a minimum membrane potential
    and :math:`\\sum z_{in}(t) z_{rec}(t)` represents the sum of all input and recurrent currents at time :math:`t`.
    We also reset the membrane potential according to reset_fn:

    .. math ::
        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    Parameters:
        tau_mem: Membrane potential time constant.
        rec_connect: An nn.Module which defines the recurrent connectivity, e.g. nn.Linear
        tau_syn: Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
        spike_threshold: Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
        spike_fn: Specify how many spikes per time step per neuron can be emitted.
        reset_fn: Specify how a neuron's membrane potential should be reset after a spike.
        surrogate_grad_fn: Choose a surrogate gradient function from sinabs.activation
        min_v_mem: Lower bound for membrane potential v_mem, clipped at every time step.
        train_alphas: When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
        shape: Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        norm_input: When True, normalise input current by tau. This helps when training time constants.
        record_states: When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.

    Shape:
        - Input: :math:`(Batch, Time, Channel, Height, Width)` or :math:`(Batch, Time, Channel)`
        - Output: Same as input.

    Attributes:
        v_mem: The membrane potential resets according to reset_fn for every spike.
        i_syn: This attribute is only available if tau_syn is not None.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: torch.Tensor = torch.tensor(1.0),
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            shape=shape,
            train_alphas=train_alphas,
            norm_input=norm_input,
            record_states=record_states,
        )
        self.rec_connect = rec_connect

    def forward(self, input_data: torch.Tensor):
        """
        Parameters:
            input_data: Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            Output data with same shape as `input_data`.
        """
        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        alpha_mem = self.alpha_mem_calculated
        alpha_syn = self.alpha_syn_calculated

        spikes, state, recordings = functional.lif_recurrent(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            spike_threshold=self.spike_threshold,
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem,
            norm_input=self.norm_input,
            rec_connect=self.rec_connect,
            record_states=self.record_states,
        )
        self.v_mem = state["v_mem"]
        self.i_syn = state["i_syn"] if alpha_syn is not None else None
        self.recordings = recordings

        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(rec_connect=self.rec_connect)
        return param_dict


class LIFSqueeze(LIF, SqueezeMixin):
    """
    :class:`~sinabs.layers.LIF` layer with 4-dimensional input (Batch*Time, Channel, Height, Width).

    Same as parent :class:`~sinabs.layers.LIF` class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.

    Shape:
        - Input: :math:`(Batch \\times Time, Channel, Height, Width)` or :math:`(Batch \\times Time, Channel)`
        - Output: Same as input.

    Attributes:
        v_mem: The membrane potential resets according to reset_fn for every spike.
        i_syn: This attribute is only available if tau_syn is not None.
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
        """Forward call wrapper that will flatten the input to and unflatten the output from the
        super class forward call."""
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
