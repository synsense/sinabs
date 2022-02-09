import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from . import functional
from .stateful_layer import StatefulLayer
from .squeeze_layer import SqueezeMixin


class LIF(StatefulLayer):
    """
    Pytorch implementation of a Leaky Integrate and Fire neuron layer.

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
    activation_fn: Callable
        a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
    threshold_low: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
    ):
        super().__init__(state_names=["v_mem", "i_syn"] if tau_syn else ["v_mem"])
        if train_alphas:
            self.alpha_mem = nn.Parameter(
                torch.exp(-1.0 / torch.as_tensor(tau_mem, dtype=torch.float32))
            )
            self.alpha_syn = (
                nn.Parameter(
                    torch.exp(-1.0 / torch.as_tensor(tau_syn, dtype=torch.float32))
                )
                if tau_syn
                else None
            )
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem, dtype=torch.float32))
            self.tau_syn = (
                nn.Parameter(torch.as_tensor(tau_syn, dtype=torch.float32))
                if tau_syn
                else None
            )
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        self.norm_input = norm_input
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        if self.train_alphas:
            return self.alpha_mem
        else:
            # we're going to always calculate this alpha parameter on CPU for 64 bit floating point precision
            original_device = self.tau_mem.device
            return torch.exp(-1.0 / self.tau_mem.to("cpu")).to(original_device)

    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        if not self.train_alphas and self.tau_syn:
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

        spikes, state = functional.lif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            norm_input=self.norm_input,
        )
        self.v_mem = state["v_mem"]
        self.i_syn = state["i_syn"] if alpha_syn else None

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
            tau_syn=-1 / torch.log(self.alpha_syn.detach_())
            if self.train_alphas
            else self.tau_syn,
            activation_fn=self.activation_fn,
            train_alphas=self.train_alphas,
            shape=self.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict


class LIFRecurrent(LIF):
    """
    Pytorch implementation of a Leaky Integrate and Fire neuron layer.

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
    activation_fn: Callable
        a torch.autograd.Function to provide forward and backward calls. Takes care of all the spiking behaviour.
    threshold_low: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            activation_fn=activation_fn,
            shape=shape,
            train_alphas=train_alphas,
            norm_input=norm_input,
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

        spikes, state = functional.lif_recurrent(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            norm_input=self.norm_input,
            rec_connect=self.rec_connect,
        )
        self.v_mem = state["v_mem"]
        self.i_syn = state["i_syn"] if alpha_syn else None

        return spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(rec_connect=self.rec_connect)
        return param_dict


class LIFSqueeze(LIF, SqueezeMixin):
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
