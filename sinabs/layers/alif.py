import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ALIFActivationFunction, MembraneSubtract, SingleSpike
from . import functional
from .stateful_layer import StatefulLayer
from .squeeze_layer import SqueezeMixin


class ALIF(StatefulLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        b0: float = 1.,
        activation_fn: Callable = None,
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
    ):
        """
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
        activation_fn: Callable
            a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        train_alphas: bool
            When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself. 
        """
        super().__init__(
            state_names = ['v_mem', 'i_syn', 'b', 'threshold'] if tau_syn else ['v_mem', 'b', 'threshold']
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_mem)))
            self.alpha_adapt = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_adapt)))
            self.alpha_syn = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_syn))) if tau_syn else None
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem))
            self.tau_adapt = nn.Parameter(torch.as_tensor(tau_adapt))
            self.tau_syn = nn.Parameter(torch.as_tensor(tau_syn)) if tau_syn else None
        self.adapt_scale = adapt_scale
        if activation_fn is None:
            self.activation_fn = ALIFActivationFunction(spike_fn=SingleSpike, reset_fn=MembraneSubtract())
        else:
            self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        self.b0 = b0
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        return self.alpha_mem if self.train_alphas else torch.exp(-1/self.tau_mem)
    
    @property
    def alpha_adapt_calculated(self):
        return self.alpha_adapt if self.train_alphas else torch.exp(-1/self.tau_adapt)
    
    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        elif not self.train_alphas and self.tau_syn:
            return torch.exp(-1/self.tau_syn)
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
        if not self.is_state_initialised() or not self.state_has_shape((batch_size, *trailing_dim)):
            self.init_state_with_shape((batch_size, *trailing_dim))
        
        alpha_mem = self.alpha_mem_calculated
        alpha_syn = self.alpha_syn_calculated
        alpha_adapt = self.alpha_adapt_calculated

        spikes, state = functional.alif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=self.adapt_scale,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            threshold_low=self.threshold_low,
            b0=self.b0,
        )
        self.threshold = state['threshold']
        self.b = state['b']
        self.v_mem = state['v_mem']

        return spikes

    def reset_states(self, randomize=False):
        super().reset_states(randomize=randomize)
        if self.is_state_initialised():
            self.threshold.fill_(self.b0)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            tau_mem=-1/torch.log(self.alpha_mem.detach_()) if self.train_alphas else self.tau_mem,
            tau_adapt=-1/torch.log(self.alpha_adapt.detach_()) if self.train_alphas else self.tau_adapt,
            tau_syn=-1/torch.log(self.alpha_syn.detach_()) if self.train_alphas else self.tau_syn,
            adapt_scale=self.adapt_scale,
            b0=self.b0,
            activation_fn=self.activation_fn,
            train_alphas=self.train_alphas,
            shape=self.v_mem.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict


class ALIFRecurrent(ALIF):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_adapt: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        b0: float = 1.,
        activation_fn: Callable = None,
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
    ):
        """
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
        activation_fn: Callable
            a torch.autograd.Function to provide forward and backward calls. Takes care of all the spiking behaviour.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        train_alphas: bool
            When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself. 
        """
        super().__init__(
            tau_mem=tau_mem,
            tau_adapt=tau_adapt,
            tau_syn=tau_syn,
            adapt_scale=adapt_scale,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            shape=shape,
            train_alphas=train_alphas,
            b0=b0,
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

        alpha_mem = self.alpha_mem_calculated
        alpha_syn = self.alpha_syn_calculated
        alpha_adapt = self.alpha_adapt_calculated

        spikes, state = functional.alif_recurrent(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=self.adapt_scale,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            threshold_low=self.threshold_low,
            rec_connect=self.rec_connect,
            b0=self.b0,
        )
        self.threshold = state['threshold']
        self.b = state['b']
        self.v_mem = state['v_mem']

        return spikes


class ALIFSqueeze(ALIF, SqueezeMixin):
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
