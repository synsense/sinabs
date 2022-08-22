import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import MultiSpike, MembraneSubtract, SingleExponential
from . import functional
from .stateful_layer import StatefulLayer
from .reshape import SqueezeMixin


class LIF(StatefulLayer):
    """
    Leaky Integrate and Fire neuron layer.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = \\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t)

        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    where :math:`\\alpha =  e^{-1/tau_{mem}}` and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    tau_mem: float
        Membrane potential time constant.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
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
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: float = 1.0,
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
        self.spike_threshold = spike_threshold
        self.spike_fn = spike_fn
        self.reset_fn = reset_fn
        self.surrogate_grad_fn = surrogate_grad_fn
        self.min_v_mem = min_v_mem
        self.train_alphas = train_alphas
        self.norm_input = norm_input
        self.record_states = record_states
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        if self.train_alphas:
            return self.alpha_mem
        else:
            return torch.exp(-1.0 / self.tau_mem)

    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        elif self.tau_syn is not None:
            # Calculate alpha with 64 bit floating point precision
            return torch.exp(-1.0 / self.tau_syn)
        else:
            return None

    @property
    def tau_mem_calculated(self):
        if self.train_alphas:
            if self.alpha_mem is None:
                return None
            else:
                return -1.0 / torch.log(self.alpha_mem)
        else:
            return self.tau_mem

    @property
    def tau_syn_calculated(self):
        if self.train_alphas:
            if self.alpha_syn is None:
                return None
            else:
                return -1.0 / torch.log(self.alpha_syn)
        else:
            return self.tau_syn

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
    """
    Leaky Integrate and Fire neuron layer with recurrent connections.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = \\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t)

        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    where :math:`\\alpha =  e^{-1/tau_{mem}}` and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    tau_mem: float
        Membrane potential time constant.
    rec_connect: torch.nn.Module
        An nn.Module which defines the recurrent connectivity, e.g. nn.Linear
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    activation_fn: Callable
        a torch.autograd.Function to provide forward and backward calls. Takes care of all the spiking behaviour.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: float = 1.0,
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
    LIF layer with 4-dimensional input (Batch*Time, Channel, Height, Width).

    Same as parent LIF class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
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
