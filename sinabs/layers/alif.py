import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import SingleSpike, MembraneSubtract, SingleExponential
from . import functional
from .stateful_layer import StatefulLayer
from .reshape import SqueezeMixin


class ALIF(StatefulLayer):
    """
    Adaptive Leaky Integrate and Fire neuron layer.

    Pytorch implementation of a Long Short Term Memory SNN (LSNN) by Bellec et al., 2018:
    https://papers.neurips.cc/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html

    Neuron dynamics in discrete time:

    .. math ::
        V(t+1) = \\alpha V(t) + (1-\\alpha) \\sum w.s(t)

        B(t+1) = b0 + \\text{adapt_scale } b(t)

        b(t+1) = \\rho b(t) + (1-\\rho) s(t)

        \\text{if } V_{mem}(t) >= B(t) \\text{, then } V_{mem} \\rightarrow V_{mem} - b0, b \\rightarrow 0

    where :math:`\\alpha = e^{-1/\\tau_{mem}}`, :math:`\\rho = e^{-1/\\tau_{adapt}}`
    and :math:`w.s(t)` is the input current for a spike s and weight w.

    By default there will not be any synaptic current dynamics used. You can specify tau_syn to apply an
    exponential decay kernel to the input:

    .. math ::
        i(t+1) = \\alpha_{syn} i(t) (1-\\alpha_{syn}) + input

    Parameters
    ----------
    tau_mem: float
        Membrane potential time constant.
    tau_adapt: float
        Spike threshold time constant.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    adapt_scale: float
        The amount that the spike threshold is bumped up for every spike, after which it decays back to the initial threshold.
    spike_threshold: float
        Set initial spike threshold. By default set to 1.0.
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
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        spike_threshold: float = 1.0,
        spike_fn: Callable = SingleSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            state_names=["v_mem", "i_syn", "b", "spike_threshold"]
            if tau_syn is not None
            else ["v_mem", "b", "spike_threshold"]
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_mem)))
            self.alpha_adapt = nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_adapt)))
            self.alpha_syn = (
                nn.Parameter(torch.exp(-1 / torch.as_tensor(tau_syn)))
                if tau_syn is not None
                else None
            )
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem))
            self.tau_adapt = nn.Parameter(torch.as_tensor(tau_adapt))
            self.tau_syn = nn.Parameter(torch.as_tensor(tau_syn)) if tau_syn else None
        self.adapt_scale = adapt_scale
        self.b0 = spike_threshold
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
        return self.alpha_mem if self.train_alphas else torch.exp(-1 / self.tau_mem)

    @property
    def alpha_adapt_calculated(self):
        return self.alpha_adapt if self.train_alphas else torch.exp(-1 / self.tau_adapt)

    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        elif not self.train_alphas and self.tau_syn is not None:
            return torch.exp(-1 / self.tau_syn)
        else:
            return None

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
        alpha_adapt = self.alpha_adapt_calculated

        spikes, state, recordings = functional.alif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=self.adapt_scale,
            state=dict(self.named_buffers()),
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem,
            b0=self.b0,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )
        self.b = state["b"]
        self.v_mem = state["v_mem"]
        self.spike_threshold = state["spike_threshold"]
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
        param_dict.update(
            tau_mem=-1 / torch.log(self.alpha_mem.detach_())
            if self.train_alphas
            else self.tau_mem,
            tau_adapt=-1 / torch.log(self.alpha_adapt.detach_())
            if self.train_alphas
            else self.tau_adapt,
            tau_syn=-1 / torch.log(self.alpha_syn.detach_())
            if self.train_alphas
            else self.tau_syn,
            adapt_scale=self.adapt_scale,
            spike_threshold=self.b0,
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            train_alphas=self.train_alphas,
            shape=self.shape,
            min_v_mem=self.min_v_mem,
            record_states=self.record_states,
        )
        return param_dict


class ALIFRecurrent(ALIF):
    """
    Adaptive Leaky Integrate and Fire neuron layer with recurrent connections.

    Pytorch implementation of a Long Short Term Memory SNN (LSNN) by Bellec et al., 2018:
    https://papers.neurips.cc/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html

    Neuron dynamics in discrete time:

    .. math ::
        V(t+1) = \\alpha V(t) + (1-\\alpha) \\sum w.s(t)

        B(t+1) = b0 + \\text{adapt_scale } b(t)

        b(t+1) = \\rho b(t) + (1-\\rho) s(t)

        \\text{if } V_{mem}(t) >= B(t) \\text{, then } V_{mem} \\rightarrow V_{mem} - b0, b \\rightarrow 0

    where :math:`\\alpha = e^{-1/\\tau_{mem}}`, :math:`\\rho = e^{-1/\\tau_{adapt}}`
    and :math:`w.s(t)` is the input current for a spike s and weight w.

    By default there will not be any synaptic current dynamics used. You can specify tau_syn to apply an
    exponential decay kernel to the input:

    .. math ::
        i(t+1) = \\alpha_{syn} i(t) (1-\\alpha_{syn}) + input

    Parameters
    ----------
    tau_mem: float
        Membrane potential time constant.
    tau_adapt: float
        Spike threshold time constant.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    adapt_scale: float
        The amount that the spike threshold is bumped up for every spike, after which it decays back to the initial threshold.
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
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        spike_threshold: float = 1.0,
        spike_fn: Callable = SingleSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        norm_input: bool = True,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_adapt=tau_adapt,
            tau_syn=tau_syn,
            adapt_scale=adapt_scale,
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
        alpha_adapt = self.alpha_adapt_calculated

        spikes, state, recordings = functional.alif_recurrent(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=self.adapt_scale,
            state=dict(self.named_buffers()),
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem,
            rec_connect=self.rec_connect,
            b0=self.b0,
            norm_input=self.norm_input,
            record_states=self.record_states,
        )
        self.b = state["b"]
        self.v_mem = state["v_mem"]
        self.spike_threshold = state["spike_threshold"]
        self.recordings = recordings

        self.firing_rate = spikes.sum() / spikes.numel()
        return spikes


class ALIFSqueeze(ALIF, SqueezeMixin):
    """
    ALIF layer with 4-dimensional input (Batch*Time, Channel, Height, Width).

    Same as parent ALIF class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
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
