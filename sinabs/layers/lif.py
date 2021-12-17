import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from . import functional
from .stateful_layer import StatefulLayer


class LIF(StatefulLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
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
        train_alphas: bool
            When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself. 
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        """
        super().__init__(
            state_names = ['v_mem', 'i_syn'] if tau_syn else ['v_mem']
        )
        if train_alphas:
            self.alpha_mem = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_mem)))
            self.alpha_syn = nn.Parameter(torch.exp(-1/torch.as_tensor(tau_syn))) if tau_syn else None
        else:
            self.tau_mem = nn.Parameter(torch.as_tensor(tau_mem))
            self.tau_syn = nn.Parameter(torch.as_tensor(tau_syn)) if tau_syn else None
        self.activation_fn = activation_fn
        self.threshold_low = threshold_low
        self.train_alphas = train_alphas
        if shape: self.init_state_with_shape(shape)

    @property
    def alpha_mem_calculated(self):
        return self.alpha_mem if self.train_alphas else torch.exp(-1/self.tau_mem)
    
    @property
    def alpha_syn_calculated(self):
        if self.train_alphas:
            return self.alpha_syn
        if not self.train_alphas and self.tau_syn:
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
        
        spikes, state = functional.lif_forward(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
        )
        self.v_mem = state['v_mem']
        self.i_syn = state['i_syn'] if alpha_syn else None
        
        return spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            tau_mem=-1/torch.log(self.alpha_mem.detach_()) if self.train_alphas else self.tau_mem,
            tau_syn=-1/torch.log(self.alpha_syn.detach_()) if self.train_alphas else self.tau_syn,
            activation_fn=self.activation_fn,
            train_alphas=self.train_alphas,
            shape=self.v_mem.shape,
            threshold_low=self.threshold_low,
        )
        return param_dict
    

class LIFRecurrent(LIF):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        rec_connect: torch.nn.Module,
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
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
        """
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            activation_fn=activation_fn,
            shape=shape,
            train_alphas=train_alphas
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
        
        spikes, state = functional.lif_recurrent(
            input_data=input_data,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=dict(self.named_buffers()),
            activation_fn=self.activation_fn,
            rec_connect=self.rec_connect,
        )
        self.v_mem = state['v_mem']
        self.i_syn = state['i_syn'] if alpha_syn else None
        
        return spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            rec_connect=self.rec_connect
        )
        return param_dict


class LIFSqueeze(LIF):
    """
    ***Deprecated class, will be removed in future release.***
    """
    def __init__(self,
                 batch_size = None,
                 num_timesteps = None,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        if not batch_size and not num_timesteps:
            raise TypeError("You need to specify either batch_size or num_timesteps.")
        if not batch_size:
            batch_size = -1 
        if not num_timesteps:
            num_timesteps = -1
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        inflated_input = input_data.reshape(self.batch_size, self.num_timesteps, *input_data.shape[1:])
        
        inflated_output = super().forward(inflated_input)
        
        return inflated_output.flatten(start_dim=0, end_dim=1)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
        )
        return param_dict