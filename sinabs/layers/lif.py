import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from sinabs.functional.lif import lif_forward
from .stateful_layer import StatefulLayer
from .recurrent_module import recurrent_class
from .pack_dims import squeeze_class


class LIF(StatefulLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        *args,
        **kwargs,
    ):
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
            a torch.autograd.Function to provide forward and backward calls. Takes care of all the spiking behaviour.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        train_alphas: bool
            When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself. 
        """
        super().__init__(
            state_names = ['v_mem', 'i_syn'] if tau_syn else ['v_mem']
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(torch.exp(-1/tau_mem))
            self.alpha_syn = nn.Parameter(torch.exp(-1/tau_syn)) if tau_syn else None
        else:
            self.tau_mem = nn.Parameter(tau_mem)
            self.tau_syn = nn.Parameter(tau_syn) if tau_syn else None
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        if shape:
            self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        return self.alpha_mem if self.train_alphas else torch.exp(-1/self.tau_mem)
    
    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        elif not self.train_alphas and self.tau_syn:
            return torch.exp(-1/self.tau_syn)
        else:
            return None

    @property
    def _param_dict(self) -> dict:
        param_dict = {}
        param_dict["activation_fn"] = self.activation_fn
        param_dict["threshold_low"] = self.threshold_low
        param_dict["train_alphas"] = self.train_alphas
        param_dict["tau_mem"] = 1/torch.log(self.alpha_mem) if self.train_alphas else self.tau_mem
        param_dict["tau_syn"] = 1/torch.log(self.alpha_syn) if self.train_alphas else self.tau_syn
        param_dict["shape"] = self.v_mem.shape
        return param_dict

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
        
        spikes, state = lif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            threshold_low=self.threshold_low,
        )
        self.v_mem = state['v_mem']
        self.i_syn = state['i_syn'] if alpha_syn else None
        
        return spikes



LIFRecurrent = recurrent_class(LIF)
LIFSqueeze = squeeze_class(LIF)
LIFRecurrentSqueeze = squeeze_class(LIFRecurrent)
