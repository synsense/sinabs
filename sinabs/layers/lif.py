from typing import Optional, Union, Callable
import torch
import torch.nn as nn
from .stateful_layer import StatefulLayer
from .recurrent_module import recurrent_class
from .pack_dims import squeeze_class
from sinabs.activation import ActivationFunction


class LIF(StatefulLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = ActivationFunction(),
        v_mem_min: Optional[float] = None,
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
            Synaptic decay time constants. If None, no synaptic dynamics are present, which is the default.
        v_mem_min: float or None
            Lower bound for membrane potential.
        """
        super().__init__(
            state_names = ['v_mem']
        )
        self.params = nn.ParameterDict({
                'tau_mem': nn.Parameter(tau_mem),
                'tau_syn': nn.Parameter(tau_syn),
        })
        self.activation_fn = activation_fn
        self.v_mem_min = v_mem_min

    @property
    def alpha_mem(self):
        return torch.exp(-1/self.params['tau_mem'])
    
    @property
    def alpha_syn(self):
        return torch.exp(-1/self.params['tau_syn'])

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

        # Ensure the neuron state are initialized
        if not self.are_states_initialised():
            self.init_states_with_shape((batch_size, *trailing_dim))

        output_spikes = []
        for step in range(time_steps):
            # Decay the membrane potential and add the input currents which are normalised by tau
            self.v_mem = self.v_mem * self.alpha_mem + (1 - self.alpha_mem) * input_current[:, step]

            # Clip membrane potential that is too low
            if self.v_mem_min:
                self.v_mem = torch.clamp(self.v_mem, min=self.v_mem_min)

            # generate spikes
            spikes = self.activation_fn(self.states())
            output_spikes.append(spikes)

        return torch.stack(output_spikes, 1)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict["tau_mem"] = self.tau_mem

        return param_dict


LIFRecurrent = recurrent_class(LIF)
LIFSqueeze = squeeze_class(LIF)
LIFRecurrentSqueeze = squeeze_class(LIFRecurrent)
