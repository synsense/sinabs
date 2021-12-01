import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from sinabs.activation import ActivationFunction
from .stateful_layer import StatefulLayer
from .recurrent_module import recurrent_class
from .pack_dims import squeeze_class


class LIF(StatefulLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        v_mem_min: Optional[float] = None,
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
        v_mem_min: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
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
        self.v_mem_min = v_mem_min
        self.train_alphas = train_alphas

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

    def forward(self, input_current: torch.Tensor):
        """
        Forward pass with given data.

        Parameters
        ----------
        input_current : torch.Tensor
            Data to be processed. Expected shape: (batch, time, ...)

        Returns
        -------
        torch.Tensor
            Output data. Same shape as `input_spikes`.
        """
        batch_size, time_steps, *trailing_dim = input_current.shape

        # Ensure the neuron states are initialized
        if not self.are_states_initialised():
            self.init_states_with_shape((batch_size, *trailing_dim))

        alpha_mem = self.alpha_mem_calculated
        alpha_syn = self.alpha_syn_calculated
        output_spikes = []
        for step in range(time_steps):
            # if t_syn was provided, we're going to use synaptic current dynamics
            if alpha_syn:
                self.i_syn = alpha_syn * self.i_syn + input_current[:, step]
            else:
                self.i_syn = input_current[:, step]
            
            # Decay the membrane potential and add the input currents which are normalised by tau
            self.v_mem = alpha_mem * self.v_mem + (1 - alpha_mem) * self.i_syn
            
            # Clip membrane potential that is too low
            if self.v_mem_min:
                self.v_mem = torch.nn.functional.relu(self.v_mem - self.v_mem_min) + self.v_mem_min

            # generate spikes and adjust v_mem
            spikes, states = self.activation_fn(dict(self.named_buffers()))
            self.v_mem = states['v_mem']
            output_spikes.append(spikes)

        return torch.stack(output_spikes, 1)


LIFRecurrent = recurrent_class(LIF)
LIFSqueeze = squeeze_class(LIF)
LIFRecurrentSqueeze = squeeze_class(LIFRecurrent)
